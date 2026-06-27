import concurrent.futures
import contextlib
import gc
import multiprocessing as mp
import random
import time

import chess
import numpy as np
import torch
from tqdm import tqdm

from encoding import board_to_tokens, canonical_square, legal_moves_by_square_pair
from evaluation import estimate_elo
from model import ChessNet
from policy import batched_policy_step
from training import train_batch

_GLOBAL_MODEL = None


def worker_init(device_type, d_model, nhead, enc_layers, heatmap_hidden, move_emb_dim):
    global _GLOBAL_MODEL
    torch.set_num_threads(1)
    _GLOBAL_MODEL = ChessNet(
        d_model=d_model,
        nhead=nhead,
        enc_layers=enc_layers,
        heatmap_hidden=heatmap_hidden,
        move_emb_dim=move_emb_dim,
    ).to(torch.device(device_type))


def play_games_batched(
    model,
    device,
    num_games=128,
    max_moves=120,
    sample_moves=15,
    temperature=1.0,
    td_lambda=0.8,
    adv_clip=3.0,
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
            temperature=temperature if ply < sample_moves else 0.0,
        )

        for idx, original_i in enumerate(active_indices):
            board, move, value = boards[original_i], moves[idx], values[idx]
            trajectories[original_i].append(
                {
                    "board_tokens": board_to_tokens(board),
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
        outcome_white = (
            1.0
            if out and out.winner == chess.WHITE
            else -1.0 if out and out.winner == chess.BLACK else 0.0
        )
        returns = [0.0] * len(trajectory)
        returns[-1] = (
            outcome_white if trajectory[-1]["turn"] == chess.WHITE else -outcome_white
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
            np.array(step["board_tokens"], dtype=np.uint8),
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
        td_lambda=td_lambda,
        adv_clip=adv_clip,
    )


def generate_self_play_data(
    model,
    total_games,
    max_moves,
    sample_moves,
    temperature,
    device,
    config,
    max_workers=None,
    executor=None,
):
    if device.type in ("cuda", "mps"):
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            return play_games_batched(
                model,
                device,
                num_games=total_games,
                max_moves=max_moves,
                sample_moves=sample_moves,
                temperature=temperature,
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
            config.move_emb_dim,
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
                config.move_emb_dim,
            ),
        )
        if use_multiprocessing
        else contextlib.nullcontext()
    )

    with executor_cm as executor:
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
                    device,
                    config,
                    max_workers=max_workers,
                    executor=executor,
                )
            )

            if (it + 1) % eval_interval == 0:
                estimate_elo(model, device, config, elo_state)
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
