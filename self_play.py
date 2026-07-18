import concurrent.futures
import contextlib
import gc
import math
import multiprocessing as mp
import os
import random
import time
from typing import Any

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
_GLOBAL_OPPONENT = None


def worker_init(
    device_type,
    d_model,
    nhead,
    enc_layers,
    heatmap_hidden,
    attn_rank,
    diffuser_hidden,
    diffuser_depth,
    diffuser_train_timesteps,
    diffuser_inference_steps,
    diffuser_fusion_enabled,
):
    global _GLOBAL_MODEL, _GLOBAL_OPPONENT
    torch.set_num_threads(1)
    _GLOBAL_MODEL, _GLOBAL_OPPONENT = (
        ChessNet(
            d_model=d_model,
            nhead=nhead,
            enc_layers=enc_layers,
            heatmap_hidden=heatmap_hidden,
            attn_rank=attn_rank,
            diffuser_hidden=diffuser_hidden,
            diffuser_depth=diffuser_depth,
            diffuser_train_timesteps=diffuser_train_timesteps,
            diffuser_inference_steps=diffuser_inference_steps,
            diffuser_fusion_enabled=diffuser_fusion_enabled,
        ).to(torch.device(device_type))
        for _ in range(2)
    )


def add_to_pool(pool, model, pool_size):
    pool.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
    if len(pool) > pool_size:
        pool.pop(0)


def elo_z_score(candidate_elo, candidate_se, reference_elo, reference_se):
    return (candidate_elo - reference_elo) / math.sqrt(
        (candidate_se or float("inf")) ** 2 + (reference_se or float("inf")) ** 2
    )


def game_over(board, ply, draw_check_interval=4):
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        return True
    return ply % draw_check_interval == 0 and board.is_game_over(claim_draw=True)


def outcome_targets(
    outcome, turn, plies, max_moves, draw_value, quick_win_bonus, return_clip
):
    if outcome.winner is None:
        return draw_value, draw_value
    if outcome.winner == turn:
        return 1.0, float(
            np.clip(
                1.0 + quick_win_bonus * max(0.0, (max_moves - plies) / max_moves),
                -return_clip,
                return_clip + quick_win_bonus,
            )
        )
    return -1.0, -1.0


def play_games_batched(
    model,
    device,
    num_games=128,
    max_moves=120,
    sample_moves=15,
    temperature=1.0,
    temperature_floor=0.25,
    adv_clip=2.0,
    draw_value=-0.15,
    quick_win_bonus=0.35,
    decisive_weight=1.5,
    return_clip=1.0,
    opponent_model=None,
    policy_candidates=None,
):
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()
    boards = [chess.Board() for _ in range(num_games)]
    learner_color = [
        chess.WHITE if random.random() < 0.5 else chess.BLACK for _ in range(num_games)
    ]
    trajectories: list[list[dict[str, Any]]] = [[] for _ in range(num_games)]
    finished = [False] * num_games

    for ply in range(max_moves):
        active = [i for i, f in enumerate(finished) if not f]
        if not active:
            break

        learner_idx = [
            i
            for i in active
            if opponent_model is None or boards[i].turn == learner_color[i]
        ]
        opponent_idx = [i for i in active if i not in learner_idx]

        if learner_idx:
            moves, values, _, log_probs = batched_policy_step(
                [boards[i] for i in learner_idx],
                model,
                device,
                temperature=temperature if ply < sample_moves else temperature_floor,
                max_candidates=policy_candidates,
            )
            for idx, original_i in enumerate(learner_idx):
                board, move, value, log_prob = (
                    boards[original_i],
                    moves[idx],
                    values[idx],
                    log_probs[idx],
                )
                trajectories[original_i].append(
                    {
                        "board_input": board_to_input(board),
                        "legal_pairs": np.array(
                            list(legal_moves_by_square_pair(board).keys()),
                            dtype=np.uint8,
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
                        "log_prob": log_prob,
                        "temperature": (
                            temperature if ply < sample_moves else temperature_floor
                        ),
                        "turn": board.turn,
                    }
                )
                board.push(move)
                if game_over(board, ply):
                    finished[original_i] = True

        if opponent_idx:
            moves, _, _, _ = batched_policy_step(
                [boards[i] for i in opponent_idx],
                opponent_model,
                device,
                temperature=temperature_floor,
                max_candidates=policy_candidates,
            )
            for idx, original_i in enumerate(opponent_idx):
                board = boards[original_i]
                board.push(moves[idx])
                if game_over(board, ply):
                    finished[original_i] = True

    raw = []
    for board, trajectory, is_finished in zip(boards, trajectories, finished):
        if not trajectory:
            continue
        outcome = board.outcome(claim_draw=True) if is_finished else None
        for step_index, step in enumerate(trajectory):
            if outcome is not None:
                value_target, policy_target = outcome_targets(
                    outcome,
                    step["turn"],
                    len(trajectory) - step_index,
                    max_moves,
                    draw_value,
                    quick_win_bonus,
                    return_clip,
                )
            else:
                value_target = policy_target = draw_value
            raw.append((step, value_target, policy_target - step["value_pred"]))

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
            decisive_weight if abs(g) == 1.0 else 1.0,
            step["log_prob"],
            step["temperature"],
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
    adv_clip,
    draw_value,
    quick_win_bonus,
    decisive_weight,
    return_clip,
    device_type,
    opponent_state_dict=None,
    policy_candidates=None,
):
    global _GLOBAL_MODEL, _GLOBAL_OPPONENT
    assert _GLOBAL_MODEL
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_type)
    _GLOBAL_MODEL.load_state_dict(state_dict)
    _GLOBAL_MODEL.eval()
    opponent = None
    if opponent_state_dict is not None:
        _GLOBAL_OPPONENT.load_state_dict(opponent_state_dict)
        _GLOBAL_OPPONENT.eval()
        opponent = _GLOBAL_OPPONENT

    return play_games_batched(
        _GLOBAL_MODEL,
        device,
        num_games=num_games,
        max_moves=max_moves,
        sample_moves=sample_moves,
        temperature=temperature,
        temperature_floor=temperature_floor,
        adv_clip=adv_clip,
        draw_value=draw_value,
        quick_win_bonus=quick_win_bonus,
        decisive_weight=decisive_weight,
        return_clip=return_clip,
        opponent_model=opponent,
        policy_candidates=policy_candidates,
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
    opponent_model=None,
    opponent_state_dict=None,
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
                adv_clip=config.self_play_adv_clip,
                draw_value=config.self_play_draw_value,
                quick_win_bonus=config.self_play_quick_win_bonus,
                decisive_weight=config.self_play_decisive_weight,
                return_clip=config.self_play_return_clip,
                opponent_model=opponent_model,
                policy_candidates=config.self_play_policy_candidates,
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
                config.self_play_adv_clip,
                config.self_play_draw_value,
                config.self_play_quick_win_bonus,
                config.self_play_decisive_weight,
                config.self_play_return_clip,
                device.type,
                opponent_state_dict,
                config.self_play_policy_candidates,
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
            config.attn_type_rank,
            config.diffuser_hidden,
            config.diffuser_depth,
            config.diffuser_train_timesteps,
            config.diffuser_inference_steps,
            config.diffuser_fusion_enabled,
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
                config.attn_type_rank,
                config.diffuser_hidden,
                config.diffuser_depth,
                config.diffuser_train_timesteps,
                config.diffuser_inference_steps,
                config.diffuser_fusion_enabled,
            ),
        )
        if use_multiprocessing
        else contextlib.nullcontext()
    )

    with executor_cm as executor:
        if "elo_ema" not in elo_state:
            estimate_elo(model, device, config, elo_state)

        elo_state["best_state"] = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        elo_state["best_scheduler_state"] = scheduler.state_dict()
        elo_state["best_elo"] = elo_state["elo_ema"]
        elo_state["best_se"] = elo_state.get("last_se")

        ref_model = ChessNet(
            d_model=config.d_model,
            nhead=config.nhead,
            enc_layers=config.enc_layers,
            heatmap_hidden=config.heatmap_hidden,
            attn_rank=config.attn_type_rank,
            diffuser_hidden=config.diffuser_hidden,
            diffuser_depth=config.diffuser_depth,
            diffuser_train_timesteps=config.diffuser_train_timesteps,
            diffuser_inference_steps=config.diffuser_inference_steps,
            diffuser_fusion_enabled=config.diffuser_fusion_enabled,
        ).to(device)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        opponent_model = ChessNet(
            d_model=config.d_model,
            nhead=config.nhead,
            enc_layers=config.enc_layers,
            heatmap_hidden=config.heatmap_hidden,
            attn_rank=config.attn_type_rank,
            diffuser_hidden=config.diffuser_hidden,
            diffuser_depth=config.diffuser_depth,
            diffuser_train_timesteps=config.diffuser_train_timesteps,
            diffuser_inference_steps=config.diffuser_inference_steps,
            diffuser_fusion_enabled=config.diffuser_fusion_enabled,
        ).to(device)
        opponent_model.eval()
        for p in opponent_model.parameters():
            p.requires_grad_(False)

        pool = [elo_state["best_state"]]

        pbar = tqdm(
            range(config.self_play_iterations), desc="Self-Play RL Optimization"
        )
        eval_interval = max(1, config.self_play_iterations // config.elo_eval_count)
        bad_evals = 0
        for it in pbar:
            opponent_state = (
                None
                if random.random() < config.self_play_pool_self_prob
                else random.choice(pool)
            )
            if opponent_state is not None and not use_multiprocessing:
                opponent_model.load_state_dict(opponent_state)

            replay.reset_rl()
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
                    opponent_model=None if opponent_state is None else opponent_model,
                    opponent_state_dict=opponent_state,
                )
            )

            if (it + 1) % config.self_play_pool_update_interval == 0:
                add_to_pool(pool, model, config.self_play_pool_size)

            if (it + 1) % config.self_play_ref_sync_interval == 0:
                ref_model.load_state_dict(model.state_dict())

            if (it + 1) % eval_interval == 0:
                _, elo_ema = estimate_elo(model, device, config, elo_state)
                z = elo_z_score(
                    elo_ema,
                    elo_state["last_se"],
                    elo_state["best_elo"],
                    elo_state["best_se"],
                )
                if z > config.self_play_promote_z:
                    elo_state["best_elo"] = elo_ema
                    elo_state["best_se"] = elo_state["last_se"]
                    elo_state["best_state"] = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    elo_state["best_scheduler_state"] = scheduler.state_dict()
                    ref_model.load_state_dict(model.state_dict())
                    add_to_pool(pool, model, config.self_play_pool_size)
                    bad_evals = 0
                elif z < -config.self_play_rollback_z:
                    bad_evals += 1
                    if bad_evals >= config.self_play_rollback_patience:
                        model.load_state_dict(elo_state["best_state"])
                        scheduler.load_state_dict(elo_state["best_scheduler_state"])
                        ref_model.load_state_dict(elo_state["best_state"])
                        opt.state.clear()
                        elo_state["elo_ema"] = elo_state["best_elo"]
                        bad_evals = 0
                else:
                    bad_evals = 0
                pbar.unpause()
            elo_postfix = (
                {"elo": f"{elo_state['elo_ema']:.0f}"} if "elo_ema" in elo_state else {}
            )

            if len(replay.rl_buf) > 0:
                losses = []
                for _ in range(config.self_play_gradient_steps):
                    batch = replay.sample_rl(config.self_play_batch_size)
                    if batch:
                        losses.append(
                            train_batch(
                                train_model,
                                opt,
                                scaler,
                                batch,
                                device,
                                ref_model=ref_model,
                                kl_coef=config.self_play_kl_coef,
                                clip_epsilon=config.self_play_clip_ratio,
                                use_diffuser=config.diffuser_fusion_enabled,
                                diffuser_steps=config.diffuser_inference_steps,
                            )
                        )
                    scheduler.step()
                if losses:
                    avg_loss, avg_p, avg_v = (
                        sum(x[i] for x in losses) / len(losses) for i in range(3)
                    )
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.3f}",
                            "policy": f"{avg_p:.3f}",
                            "value": f"{avg_v:.3f}",
                            "rl_buf": len(replay.rl_buf),
                            **elo_postfix,
                        }
                    )
                else:
                    pbar.set_postfix({"rl_buf": len(replay.rl_buf), **elo_postfix})
            else:
                pbar.set_postfix({"rl_buf": len(replay.rl_buf), **elo_postfix})

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        final_elo, _ = estimate_elo(model, device, config, elo_state)
        z = elo_z_score(
            final_elo,
            elo_state["last_se"],
            elo_state["best_elo"],
            elo_state["best_se"],
        )
        if z < -config.self_play_rollback_z:
            model.load_state_dict(elo_state["best_state"])


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
