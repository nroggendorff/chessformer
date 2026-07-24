import concurrent.futures
import contextlib
import gc
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

import chess
from config import amp_dtype
from encoding import board_to_input, legal_moves_by_square_pair
from evaluation import estimate_elo
from model import ChessNet
from training import train_batch
from tree_search import choose_move, run_mcts, visit_policy_pairs

_GLOBAL_MODEL = None
_GLOBAL_OPPONENT = None


def worker_init(device_type, d_model, nhead, enc_layers, heatmap_hidden, attn_rank):
    global _GLOBAL_MODEL, _GLOBAL_OPPONENT
    gc.set_threshold(100000, 50, 50)
    torch.set_num_threads(1)
    _GLOBAL_MODEL, _GLOBAL_OPPONENT = (
        ChessNet(
            d_model=d_model,
            nhead=nhead,
            enc_layers=enc_layers,
            heatmap_hidden=heatmap_hidden,
            attn_rank=attn_rank,
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
    return board.outcome(claim_draw=ply % draw_check_interval == 0) is not None


def play_games_batched(
    model,
    device,
    num_games=128,
    max_moves=120,
    sample_moves=15,
    temperature=1.0,
    temperature_floor=0.1,
    decisive_weight=1.5,
    mcts_simulations=200,
    opponent_mcts_simulations=100,
    sims_per_wave=8,
    target_batch_size=None,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    root_noise_frac=0.25,
    opponent_model=None,
):
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()
    boards = [chess.Board() for _ in range(num_games)]
    learner_color = [
        chess.WHITE if random.random() < 0.5 else chess.BLACK for _ in range(num_games)
    ]
    trajectories: list[list[dict]] = [[] for _ in range(num_games)]
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
        temperature_now = temperature if ply < sample_moves else temperature_floor

        if learner_idx:
            roots = run_mcts(
                [boards[i] for i in learner_idx],
                model,
                device,
                num_simulations=mcts_simulations,
                sims_per_wave=sims_per_wave,
                target_batch_size=target_batch_size,
                c_puct=c_puct,
                add_root_noise=True,
                root_dirichlet_alpha=dirichlet_alpha,
                root_noise_frac=root_noise_frac,
            )
            for original_i, root in zip(learner_idx, roots):
                board = boards[original_i]
                move = choose_move(root, temperature_now)
                if move is None:
                    finished[original_i] = True
                    continue
                policy_pairs = visit_policy_pairs(root)
                trajectories[original_i].append(
                    {
                        "board_input": board_to_input(
                            board, legal_moves=root.legal_moves
                        ),
                        "legal_pairs": np.array(
                            list(
                                legal_moves_by_square_pair(
                                    board, legal_moves=root.legal_moves
                                ).keys()
                            ),
                            dtype=np.uint8,
                        ),
                        "policy_pairs": np.array(
                            list(policy_pairs.keys()), dtype=np.uint8
                        ).reshape(-1, 2),
                        "policy_probs": np.array(
                            list(policy_pairs.values()), dtype=np.float32
                        ),
                        "turn": board.turn,
                    }
                )
                board.push(move)
                if game_over(board, ply):
                    finished[original_i] = True

        if opponent_idx:
            roots = run_mcts(
                [boards[i] for i in opponent_idx],
                opponent_model,
                device,
                num_simulations=opponent_mcts_simulations,
                sims_per_wave=sims_per_wave,
                target_batch_size=target_batch_size,
                c_puct=c_puct,
                add_root_noise=False,
            )
            for original_i, root in zip(opponent_idx, roots):
                board = boards[original_i]
                move = choose_move(root, temperature_floor)
                if move is None:
                    finished[original_i] = True
                    continue
                board.push(move)
                if game_over(board, ply):
                    finished[original_i] = True

    samples = []
    for board, trajectory, is_finished in zip(boards, trajectories, finished):
        if not trajectory:
            continue
        winner = board.outcome(claim_draw=True).winner if is_finished else None
        policy_weight = decisive_weight if winner is not None else 1.0
        value_weight = policy_weight if is_finished else 0.0
        for step in trajectory:
            value_target = (
                0.0
                if winner is None
                else float(1.0 if winner == step["turn"] else -1.0)
            )
            samples.append(
                (
                    np.array(step["board_input"], dtype=np.uint8),
                    step["legal_pairs"],
                    step["policy_pairs"],
                    step["policy_probs"],
                    value_target,
                    policy_weight,
                    value_weight,
                )
            )

    return samples


def worker_play_games(
    state_dict,
    seed,
    num_games,
    max_moves,
    sample_moves,
    temperature,
    temperature_floor,
    decisive_weight,
    mcts_simulations,
    opponent_mcts_simulations,
    sims_per_wave,
    target_batch_size,
    c_puct,
    dirichlet_alpha,
    root_noise_frac,
    device_type,
    opponent_state_dict=None,
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
        decisive_weight=decisive_weight,
        mcts_simulations=mcts_simulations,
        opponent_mcts_simulations=opponent_mcts_simulations,
        sims_per_wave=sims_per_wave,
        target_batch_size=target_batch_size,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        root_noise_frac=root_noise_frac,
        opponent_model=opponent,
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
    use_multiprocessing,
    max_workers=None,
    executor=None,
    opponent_model=None,
    opponent_state_dict=None,
):
    if not use_multiprocessing:
        with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
            return play_games_batched(
                model,
                device,
                num_games=total_games,
                max_moves=max_moves,
                sample_moves=sample_moves,
                temperature=temperature,
                temperature_floor=temperature_floor,
                decisive_weight=config.self_play_decisive_weight,
                mcts_simulations=config.self_play_mcts_simulations,
                opponent_mcts_simulations=config.self_play_opponent_mcts_simulations,
                sims_per_wave=config.mcts_sims_per_wave,
                target_batch_size=config.mcts_target_batch_size,
                c_puct=config.mcts_c_puct,
                dirichlet_alpha=config.mcts_dirichlet_alpha,
                root_noise_frac=config.mcts_root_noise_frac,
                opponent_model=opponent_model,
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
                config.self_play_decisive_weight,
                config.self_play_mcts_simulations,
                config.self_play_opponent_mcts_simulations,
                config.mcts_sims_per_wave,
                max(
                    config.mcts_sims_per_wave,
                    config.mcts_target_batch_size // max_workers,
                ),
                config.mcts_c_puct,
                config.mcts_dirichlet_alpha,
                config.mcts_root_noise_frac,
                "cpu",
                opponent_state_dict,
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
            "cpu",
            config.d_model,
            config.nhead,
            config.enc_layers,
            config.heatmap_hidden,
            config.attn_type_rank,
        ),
        max_tasks_per_child=config.self_play_worker_max_tasks,
    ) as fresh_executor:
        futures = submit(fresh_executor)
        return [s for f in concurrent.futures.as_completed(futures) for s in f.result()]


def run_self_play(
    model, train_model, opt, scaler, scheduler, replay, device, config, elo_state
):
    use_multiprocessing = (config.self_play_max_workers or 1) > 1
    max_workers = (
        min(config.self_play_max_workers, config.self_play_games_per_iter)
        if use_multiprocessing
        else None
    )

    print(
        f"Training on {device.type}; generating self-play games with "
        + (
            f"{max_workers} CPU worker processes"
            if use_multiprocessing
            else "a single batched pass"
        )
    )

    executor_cm = (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=worker_init,
            initargs=(
                "cpu",
                config.d_model,
                config.nhead,
                config.enc_layers,
                config.heatmap_hidden,
                config.attn_type_rank,
            ),
            max_tasks_per_child=config.self_play_worker_max_tasks,
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

        opponent_model = ChessNet(
            d_model=config.d_model,
            nhead=config.nhead,
            enc_layers=config.enc_layers,
            heatmap_hidden=config.heatmap_hidden,
            attn_rank=config.attn_type_rank,
        ).to(device)
        opponent_model.eval()
        for p in opponent_model.parameters():
            p.requires_grad_(False)

        pool = [elo_state["best_state"]]

        pbar = tqdm(
            range(config.self_play_iterations), desc="Self-Play RL Optimization"
        )
        eval_interval = max(
            1, config.self_play_iterations // config.self_play_eval_count
        )
        bad_evals = 0
        for it in pbar:
            opponent_state = (
                None
                if random.random() < config.self_play_pool_self_prob
                else random.choice(pool)
            )
            if opponent_state is not None and not use_multiprocessing:
                opponent_model.load_state_dict(opponent_state)

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
                    use_multiprocessing,
                    max_workers=max_workers,
                    executor=executor,
                    opponent_model=None if opponent_state is None else opponent_model,
                    opponent_state_dict=opponent_state,
                )
            )

            if (it + 1) % config.self_play_pool_update_interval == 0:
                add_to_pool(pool, model, config.self_play_pool_size)

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
                    add_to_pool(pool, model, config.self_play_pool_size)
                    bad_evals = 0
                elif z < -config.self_play_rollback_z:
                    bad_evals += 1
                    if bad_evals >= config.self_play_rollback_patience:
                        model.load_state_dict(elo_state["best_state"])
                        scheduler.load_state_dict(elo_state["best_scheduler_state"])
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
                            train_batch(train_model, opt, scaler, batch, device)
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
