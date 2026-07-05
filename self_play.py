import concurrent.futures
import contextlib
import gc
import multiprocessing as mp
import os
import random
import time

import chess
import numpy as np
import torch
from tqdm import tqdm

from config import amp_dtype
from encoding import board_to_input, canonical_square, legal_moves_by_square_pair
from evaluation import estimate_elo
from model import ChessNet
from policy import batched_policy_step
from training import train_batch

_GLOBAL_MODEL = None


def worker_init(device_type, d_model, nhead, enc_layers, heatmap_hidden):
    global _GLOBAL_MODEL
    torch.set_num_threads(1)
    _GLOBAL_MODEL = ChessNet(
        d_model=d_model,
        nhead=nhead,
        enc_layers=enc_layers,
        heatmap_hidden=heatmap_hidden,
    ).to(torch.device(device_type))


def play_games_batched(
    model,
    device,
    num_games=128,
    max_moves=120,
    sample_moves=15,
    temperature=1.0,
    temperature_floor=0.25,
    td_lambda=0.8,
    adv_clip=2.0,
):
    model.eval()
    boards = [chess.Board() for _ in range(num_games)]
    trajectories = [[] for _ in range(num_games)]
    finished = [False] * num_games

    for ply in range(max_moves):
        active_indices = [i for i, f in enumerate(finished) if not f]
        if not active_indices:
            break

        moves, values, _ = batched_policy_step(
            [boards[i] for i in active_indices],
            model,
            device,
            temperature=temperature if ply < sample_moves else temperature_floor,
        )

        for idx, original_i in enumerate(active_indices):
            board, move, value = boards[original_i], moves[idx], values[idx]
            trajectories[original_i].append(
                {
                    "board_input": board_to_input(board),
                    "legal_pairs": np.array(
                        list(legal_moves_by_square_pair(board).keys()), dtype=np.uint8
                    ),
                    "policy_pair": np.array(
                        [
                            (
                                canonical_square(move.from_square, board),
                                canonical_square(move.to_square, board),
                            )
                        ],
                        dtype=np.uint8,
                    ),
                    "value_pred": value,
                    "turn": board.turn,
                }
            )
            board.push(move)
            if board.is_game_over(claim_draw=True):
                finished[original_i] = True

    raw = []
    for board, trajectory in zip(boards, trajectories):
        if not trajectory:
            continue
        out = board.outcome(claim_draw=True)
        returns = [0.0] * len(trajectory)
        if out is None:
            returns[-1] = trajectory[-1]["value_pred"]
        else:
            outcome_white = (
                1.0
                if out.winner == chess.WHITE
                else -1.0 if out.winner == chess.BLACK else 0.0
            )
            returns[-1] = (
                outcome_white
                if trajectory[-1]["turn"] == chess.WHITE
                else -outcome_white
            )
        for t in range(len(trajectory) - 2, -1, -1):
            bootstrap = -trajectory[t + 1]["value_pred"]
            next_return = -returns[t + 1]
            returns[t] = (1 - td_lambda) * bootstrap + td_lambda * next_return
        raw.extend(
            (step, g, g - step["value_pred"]) for step, g in zip(trajectory, returns)
        )

    if not raw:
        return []

    advantages = np.array([a for _, _, a in raw], dtype=np.float32)
    normalized = np.clip(
        (advantages - advantages.mean()) / (advantages.std() + 1e-6),
        -adv_clip,
        adv_clip,
    )

    return [
        (
            np.array(step["board_input"], dtype=np.uint8),
            step["legal_pairs"],
            step["policy_pair"],
            np.array([1.0], dtype=np.float32),
            g,
            float(w),
            1.0,
        )
        for (step, g, _), w in zip(raw, normalized)
    ]


def worker_play_games(
    state_dict,
    seed,
    num_games,
    max_moves,
    sample_moves,
    temperature,
    temperature_floor,
    td_lambda,
    adv_clip,
    device_type,
):
    global _GLOBAL_MODEL
    assert _GLOBAL_MODEL
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_type)
    _GLOBAL_MODEL.load_state_dict(state_dict)
    _GLOBAL_MODEL.eval()

    return play_games_batched(
        _GLOBAL_MODEL,
        device,
        num_games=num_games,
        max_moves=max_moves,
        sample_moves=sample_moves,
        temperature=temperature,
        temperature_floor=temperature_floor,
        td_lambda=td_lambda,
        adv_clip=adv_clip,
    )


def generate_self_play_data(
    model,
    total_games,
    max_moves,
    sample_moves,
    temperature,
    temperature_floor,
    device,
    config,
    max_workers=None,
    executor=None,
):
    if device.type in ("cuda", "mps"):
        with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
            return play_games_batched(
                model,
                device,
                num_games=total_games,
                max_moves=max_moves,
                sample_moves=sample_moves,
                temperature=temperature,
                temperature_floor=temperature_floor,
                td_lambda=config.self_play_td_lambda,
                adv_clip=config.self_play_adv_clip,
            )

    max_workers = min(max_workers or mp.cpu_count(), total_games)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    counts = [total_games // max_workers] * max_workers
    for i in range(total_games % max_workers):
        counts[i] += 1
    base_seed = time.time_ns() % (2**32 - len(counts))

    def submit(pool):
        return [
            pool.submit(
                worker_play_games,
                state_dict,
                base_seed + i,
                count,
                max_moves,
                sample_moves,
                temperature,
                temperature_floor,
                config.self_play_td_lambda,
                config.self_play_adv_clip,
                device.type,
            )
            for i, count in enumerate(counts)
        ]

    if executor is not None:
        futures = submit(executor)
        return [s for f in concurrent.futures.as_completed(futures) for s in f.result()]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=worker_init,
        initargs=(
            device.type,
            config.d_model,
            config.nhead,
            config.enc_layers,
            config.heatmap_hidden,
        ),
    ) as fresh_executor:
        futures = submit(fresh_executor)
        return [s for f in concurrent.futures.as_completed(futures) for s in f.result()]


def run_self_play(
    model, train_model, opt, scaler, scheduler, replay, device, config, elo_state
):
    use_multiprocessing = device.type not in ("cuda", "mps")
    max_workers = (
        min(config.max_workers, config.self_play_games_per_iter)
        if use_multiprocessing
        else None
    )

    print(
        f"Starting self-play on {device.type} with "
        + (
            f"{max_workers} CPU worker processes"
            if use_multiprocessing
            else "a single batched GPU pass"
        )
    )

    executor_cm = (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=worker_init,
            initargs=(
                device.type,
                config.d_model,
                config.nhead,
                config.enc_layers,
                config.heatmap_hidden,
            ),
        )
        if use_multiprocessing
        else contextlib.nullcontext()
    )

    with executor_cm as executor:
        elo_state["best_state"] = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        elo_state["best_elo"] = elo_state.get("elo_ema", float("-inf"))

        ref_model = ChessNet(
            d_model=config.d_model,
            nhead=config.nhead,
            enc_layers=config.enc_layers,
            heatmap_hidden=config.heatmap_hidden,
        ).to(device)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        pbar = tqdm(
            range(config.self_play_iterations), desc="Self-Play RL Optimization"
        )
        eval_interval = max(1, config.self_play_iterations // config.elo_eval_count)
        for it in pbar:
            replay.extend_rl(
                generate_self_play_data(
                    model,
                    config.self_play_games_per_iter,
                    config.self_play_max_moves,
                    config.self_play_sample_moves,
                    config.self_play_temperature,
                    config.self_play_temperature_floor,
                    device,
                    config,
                    max_workers=max_workers,
                    executor=executor,
                )
            )

            if (it + 1) % eval_interval == 0:
                _, elo_ema = estimate_elo(model, device, config, elo_state)
                if elo_ema >= elo_state["best_elo"]:
                    elo_state["best_elo"] = elo_ema
                    elo_state["best_state"] = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    ref_model.load_state_dict(model.state_dict())
                elif elo_state["best_elo"] - elo_ema > config.self_play_rollback_margin:
                    model.load_state_dict(elo_state["best_state"])
                    opt.state.clear()
                pbar.unpause()
            elo_postfix = (
                {"elo": f"{elo_state['elo_ema']:.0f}"} if "elo_ema" in elo_state else {}
            )

            if len(replay.pretrain_buf) > 0 or len(replay.rl_buf) > 0:
                losses = []
                for _ in range(config.self_play_gradient_steps):
                    losses.append(
                        train_batch(
                            train_model,
                            opt,
                            scaler,
                            replay.sample(
                                config.self_play_batch_size,
                                mix_ratio=config.self_play_mix_ratio,
                            ),
                            device,
                            ref_model=ref_model,
                            kl_coef=config.self_play_kl_coef,
                        )
                    )
                    scheduler.step()
                avg_loss, avg_p, avg_v = (
                    sum(x[i] for x in losses) / len(losses) for i in range(3)
                )
                pbar.set_postfix(
                    loss=f"{avg_loss:.3f}",
                    policy=f"{avg_p:.3f}",
                    value=f"{avg_v:.3f}",
                    rl_buf=len(replay.rl_buf),
                    **elo_postfix,
                )
            else:
                pbar.set_postfix(rl_buf=len(replay.rl_buf), **elo_postfix)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()


if __name__ == "__main__":
    from config import (
        Config,
        build_model,
        build_optimizer,
        build_scaler,
        build_scheduler,
        default_checkpoint_path,
        get_device,
    )
    from model import save_checkpoint
    from replay_buffer import DualRingBuffer

    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()
    print(
        f"Resuming self-play from {checkpoint_path}"
        if os.path.exists(checkpoint_path)
        else "Starting self-play from a randomly initialized model"
    )
    model, train_model = build_model(config, device, checkpoint_path)
    opt = build_optimizer(model, config)
    scaler = build_scaler(device)
    scheduler = build_scheduler(
        opt, config.self_play_iterations * config.self_play_gradient_steps
    )
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )

    run_self_play(
        model, train_model, opt, scaler, scheduler, replay, device, config, {}
    )
    save_checkpoint(model, checkpoint_path)
