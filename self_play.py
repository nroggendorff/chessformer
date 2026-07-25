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

from config import amp_dtype
from encoding import INPUT_SIZE
from evaluation import estimate_elo
from model import ChessNet
from self_play_game import play_games_batched
from self_play_workers import (
    calibrate_self_play_workers,
    worker_init,
    worker_play_games,
)
from training import train_batch


def warmup_train_model(model, train_model, opt, scaler, config, device):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    samples = [
        (
            np.zeros(INPUT_SIZE, dtype=np.uint8),
            np.array([[0, 1]], dtype=np.uint8),
            np.array([[0, 1]], dtype=np.uint8),
            np.array([1.0], dtype=np.float32),
            0.0,
            1.0,
            1.0,
        )
        for _ in range(config.self_play_batch_size)
    ]
    train_batch(train_model, opt, scaler, samples, device)
    model.load_state_dict(state)
    opt.state.clear()


def add_to_pool(pool, model, pool_size):
    pool.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
    if len(pool) > pool_size:
        pool.pop(0)


def elo_z_score(candidate_elo, candidate_se, reference_elo, reference_se):
    return (candidate_elo - reference_elo) / math.sqrt(
        (candidate_se or float("inf")) ** 2 + (reference_se or float("inf")) ** 2
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
    chunk = min(config.self_play_chunk_games, total_games)
    counts = [chunk] * (total_games // chunk) + (
        [total_games % chunk] if total_games % chunk else []
    )
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
                config.mcts_target_batch_size,
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
    warmup_train_model(model, train_model, opt, scaler, config, device)
    use_multiprocessing = (config.self_play_max_workers or 1) > 1
    max_workers = (
        min(
            calibrate_self_play_workers(config) if use_multiprocessing else 1,
            config.self_play_games_per_iter,
        )
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
