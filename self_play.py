import concurrent.futures
import contextlib
import gc
import math
import multiprocessing as mp
import os
import random
import time

import chess.engine
import numpy as np
import torch
from tqdm import tqdm

from config import amp_dtype, build_scheduler, set_optimizer_lr
from encoding import INPUT_SIZE
from evaluation import (
    adjudicate_unresolved,
    binomial_z_score,
    clamp_uci_elo,
    estimate_elo,
    opening_moves_for_game,
)
from model import ChessNet
from self_play_game import play_games_batched
from self_play_workers import (
    calibrate_self_play_workers,
    worker_init,
    worker_play_games,
)
from state_utils import load_state
from training import train_batch


def warmup_train_model(model, train_model, opt, scaler, config, device):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    samples = [
        (
            np.zeros(INPUT_SIZE, dtype=np.int64),
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


def _collect_worker_results(futures):
    samples = []
    stats = {
        "unresolved_positions": [],
        "games": 0,
        "decisive": 0,
        "drawn": 0,
        "unresolved": 0,
        "learner_wins": 0,
        "opponent_wins": 0,
    }
    for f in concurrent.futures.as_completed(futures):
        chunk_samples, chunk_stats = f.result()
        samples.extend(chunk_samples)
        for key in stats:
            stats[key] += chunk_stats[key]
    return samples, stats


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
    stockfish_engine=None,
    stockfish_path=None,
    stockfish_movetime=0.1,
    add_root_noise=True,
    value_smoothing=0.0,
    record_trajectory=True,
    mcts_simulations=None,
    opponent_mcts_simulations=None,
    opening_moves_per_game=None,
):
    mcts_simulations = mcts_simulations or config.self_play_mcts_simulations
    opponent_mcts_simulations = (
        opponent_mcts_simulations or config.self_play_opponent_mcts_simulations
    )
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
                timeout_value_weight=config.self_play_timeout_value_weight,
                mcts_simulations=mcts_simulations,
                opponent_mcts_simulations=opponent_mcts_simulations,
                sims_per_wave=config.mcts_sims_per_wave,
                target_batch_size=config.mcts_target_batch_size,
                max_batch_size=config.mcts_max_batch_size,
                c_puct=config.mcts_c_puct,
                fpu_reduction=config.mcts_fpu_reduction,
                dirichlet_alpha=config.mcts_dirichlet_alpha,
                root_noise_frac=config.mcts_root_noise_frac,
                opponent_model=opponent_model,
                stockfish_engine=stockfish_engine,
                stockfish_movetime=stockfish_movetime,
                resign_threshold=config.self_play_resign_threshold,
                resign_streak=config.self_play_resign_streak,
                add_root_noise=add_root_noise,
                value_smoothing=value_smoothing,
                record_trajectory=record_trajectory,
                include_policy_q_threshold=config.self_play_include_policy_q_threshold,
                opening_moves_per_game=opening_moves_per_game,
                target_beta=config.self_play_target_beta,
            )

    max_workers = min(max_workers or mp.cpu_count(), total_games)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    chunk = max(
        1, min(config.self_play_chunk_games, math.ceil(total_games / max_workers))
    )
    counts = [chunk] * (total_games // chunk) + (
        [total_games % chunk] if total_games % chunk else []
    )
    base_seed = time.time_ns() % (2**32 - len(counts))
    offsets = [sum(counts[:i]) for i in range(len(counts))]

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
                config.self_play_timeout_value_weight,
                mcts_simulations,
                opponent_mcts_simulations,
                config.mcts_sims_per_wave,
                config.mcts_target_batch_size,
                config.mcts_max_batch_size,
                config.mcts_c_puct,
                config.mcts_dirichlet_alpha,
                config.mcts_root_noise_frac,
                device.type,
                opponent_state_dict,
                stockfish_path,
                config.self_play_stockfish_elo,
                stockfish_movetime,
                config.self_play_resign_threshold,
                config.self_play_resign_streak,
                add_root_noise,
                value_smoothing,
                record_trajectory,
                config.self_play_include_policy_q_threshold,
                (
                    opening_moves_per_game[offset : offset + count]
                    if opening_moves_per_game is not None
                    else None
                ),
                config.self_play_target_beta,
                config.mcts_fpu_reduction,
            )
            for i, (count, offset) in enumerate(zip(counts, offsets))
        ]

    if executor is not None:
        return _collect_worker_results(submit(executor))

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
        max_tasks_per_child=config.self_play_worker_max_tasks,
    ) as fresh_executor:
        return _collect_worker_results(submit(fresh_executor))


def head_to_head_score(
    model,
    opponent_model,
    opponent_state,
    games,
    max_moves,
    device,
    config,
    use_multiprocessing,
    max_workers=None,
    executor=None,
):
    if not use_multiprocessing:
        load_state(opponent_model, opponent_state)
    _, stats = generate_self_play_data(
        model,
        games,
        max_moves,
        config.self_play_h2h_sample_moves,
        config.self_play_h2h_opening_temperature,
        config.self_play_temperature_floor,
        device,
        config,
        use_multiprocessing,
        max_workers=max_workers,
        executor=executor,
        opponent_model=opponent_model,
        opponent_state_dict=opponent_state,
        add_root_noise=False,
        record_trajectory=False,
        mcts_simulations=config.h2h_mcts_simulations,
        opponent_mcts_simulations=config.h2h_mcts_simulations,
        opening_moves_per_game=[
            opening_moves_for_game(i, plies=6) for i in range(games)
        ],
    )
    if config.self_play_h2h_adjudicate and stats["unresolved_positions"]:
        try:
            scores = adjudicate_unresolved(
                config.stockfish_path,
                stats["unresolved_positions"],
                config.elo_eval_adjudication_depth,
                config.self_play_h2h_adjudication_margin,
                max_workers=config.max_workers,
            )
        except (FileNotFoundError, chess.engine.EngineError) as error:
            print(
                f"head-to-head adjudication unavailable, discarding timeouts: {error}"
            )
            scores = []
        for score in scores:
            stats["unresolved"] -= 1
            if score > 0.5:
                stats["learner_wins"] += 1
                stats["decisive"] += 1
            elif score < 0.5:
                stats["opponent_wins"] += 1
                stats["decisive"] += 1
            else:
                stats["drawn"] += 1
    return stats


def run_self_play(
    model, train_model, opt, scaler, scheduler, replay, device, config, elo_state
):
    warmup_train_model(model, train_model, opt, scaler, config, device)
    gc.set_threshold(100000, 50, 50)
    use_multiprocessing = (config.self_play_max_workers or 1) > 1
    max_workers = (
        min(
            calibrate_self_play_workers(config, device) if use_multiprocessing else 1,
            config.self_play_games_per_iter,
        )
        if use_multiprocessing
        else None
    )

    print(
        f"Training on {device.type}; generating self-play games with "
        + (
            f"{max_workers} {device.type} worker processes"
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
                device.type,
                config.d_model,
                config.nhead,
                config.enc_layers,
                config.heatmap_hidden,
            ),
            max_tasks_per_child=config.self_play_worker_max_tasks,
        )
        if use_multiprocessing
        else contextlib.nullcontext()
    )

    with executor_cm as executor, contextlib.ExitStack() as stack:
        if "elo_ema" not in elo_state:
            estimate_elo(model, device, config, elo_state)

        elo_state["best_state"] = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        best_elo_state = {
            k: elo_state[k]
            for k in ("elo_ema", "last_elo", "last_se")
            if k in elo_state
        }

        opponent_model = ChessNet(
            d_model=config.d_model,
            nhead=config.nhead,
            enc_layers=config.enc_layers,
            heatmap_hidden=config.heatmap_hidden,
        ).to(device)
        opponent_model.eval()
        for p in opponent_model.parameters():
            p.requires_grad_(False)

        pool = [elo_state["best_state"]]
        stockfish_engine = None
        stockfish_available = config.self_play_stockfish_prob > 0
        if stockfish_available and not use_multiprocessing:
            try:
                stockfish_engine = stack.enter_context(
                    chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
                )
                stockfish_engine.configure(
                    {
                        "UCI_LimitStrength": True,
                        "UCI_Elo": clamp_uci_elo(
                            stockfish_engine, config.self_play_stockfish_elo
                        ),
                    }
                )
            except (FileNotFoundError, chess.engine.EngineError) as error:
                print(f"Stockfish self-play opponents disabled: {error}")
                stockfish_available = False

        pbar = tqdm(
            range(config.self_play_iterations),
            desc="Self-Play RL Optimization",
            smoothing=0.1,
        )
        eval_interval = max(
            1, round(config.self_play_iterations / config.self_play_eval_count)
        )
        bad_evals = 0
        promote_streak = 0
        last_elo_iter = 0
        start_elo = elo_state["elo_ema"]

        def rollback_to_best(it):
            nonlocal scheduler
            model.load_state_dict(elo_state["best_state"])
            opt.state.clear()
            set_optimizer_lr(opt, config.self_play_lr)
            scheduler = build_scheduler(
                opt,
                config.self_play_gradient_steps
                * max(1, config.self_play_iterations - it),
            )
            replay.reset_rl()
            elo_state.update(best_elo_state)

        for it in pbar:
            roll = random.random()
            self_threshold = config.self_play_pool_self_prob
            anchor_threshold = self_threshold + config.self_play_anchor_prob
            stockfish_threshold = anchor_threshold + config.self_play_stockfish_prob
            use_stockfish = (
                stockfish_available and anchor_threshold <= roll < stockfish_threshold
            )
            opponent_state = (
                None
                if roll < self_threshold or use_stockfish
                else (
                    elo_state["best_state"]
                    if roll < anchor_threshold
                    else random.choice(pool)
                )
            )
            if opponent_state is not None and not use_multiprocessing:
                opponent_model.load_state_dict(opponent_state)

            samples, sp_stats = generate_self_play_data(
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
                stockfish_engine=stockfish_engine if use_stockfish else None,
                stockfish_path=(
                    config.stockfish_path
                    if use_stockfish and use_multiprocessing
                    else None
                ),
                stockfish_movetime=config.self_play_stockfish_movetime,
                value_smoothing=config.self_play_value_smoothing,
            )
            replay.extend_rl(samples)

            if (it + 1) % config.self_play_pool_update_interval == 0:
                add_to_pool(pool, model, config.self_play_pool_size)

            if (it + 1) % eval_interval == 0:
                h2h = head_to_head_score(
                    model,
                    opponent_model,
                    elo_state["best_state"],
                    config.self_play_h2h_games,
                    config.self_play_max_moves,
                    device,
                    config,
                    use_multiprocessing,
                    max_workers=max_workers,
                    executor=executor,
                )
                z = binomial_z_score(
                    h2h["learner_wins"],
                    h2h["drawn"],
                    h2h["games"] - h2h["unresolved"],
                )
                record = f"{h2h['learner_wins']}-{h2h['opponent_wins']}-{h2h['drawn']}"
                if z > config.self_play_promote_z:
                    promote_streak += 1
                    bad_evals = 0
                    if promote_streak < config.self_play_promote_confirm:
                        pbar.write(
                            f"[iter {it + 1}] passed eval: scored {record} vs best "
                            f"(z={z:.2f}, confirm {promote_streak}/{config.self_play_promote_confirm})"
                        )
                    else:
                        elo_state["best_state"] = {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        }
                        if (
                            it + 1 - last_elo_iter
                            >= config.self_play_elo_refresh_interval
                        ):
                            _, elo_state["elo_ema"] = estimate_elo(
                                model, device, config, elo_state
                            )
                            last_elo_iter = it + 1
                        best_elo_state = {
                            k: elo_state[k]
                            for k in ("elo_ema", "last_elo", "last_se")
                            if k in elo_state
                        }
                        start_elo = best_elo_state["elo_ema"]
                        pbar.write(
                            f"[iter {it + 1}] promoted: scored {record} vs best "
                            f"(z={z:.2f}, confirmed {promote_streak}/{config.self_play_promote_confirm}, "
                            f"elo_ema {elo_state['elo_ema']:.0f})"
                        )
                        add_to_pool(pool, model, config.self_play_pool_size)
                        promote_streak = 0
                elif z < -config.self_play_rollback_z:
                    promote_streak = 0
                    bad_evals += 1
                    if it + 1 - last_elo_iter >= config.self_play_elo_refresh_interval:
                        _, elo_state["elo_ema"] = estimate_elo(
                            model, device, config, elo_state
                        )
                        last_elo_iter = it + 1
                    pbar.write(
                        f"[iter {it + 1}] bad eval: scored {record} vs best "
                        f"(z={z:.2f}, bad_evals={bad_evals}/{config.self_play_rollback_patience}, "
                        f"elo_ema {elo_state['elo_ema']:.0f})"
                    )
                    if bad_evals >= config.self_play_rollback_patience:
                        pbar.write(f"[iter {it + 1}] rolling back to best")
                        rollback_to_best(it)
                        bad_evals = 0
                else:
                    bad_evals = 0

                if (
                    "elo_ema" in elo_state
                    and elo_state["elo_ema"]
                    < start_elo - config.self_play_elo_drop_rollback
                ):
                    pbar.write(
                        f"[iter {it + 1}] elo_ema dropped "
                        f"{start_elo - elo_state['elo_ema']:.0f} below start "
                        f"({start_elo:.0f}); rolling back to best"
                    )
                    rollback_to_best(it)
                    bad_evals = 0
                    promote_streak = 0
            elo_postfix = (
                {"elo": f"{elo_state['elo_ema']:.0f}"} if "elo_ema" in elo_state else {}
            )
            finish_postfix = {
                "resolved": (
                    f"{(sp_stats['decisive'] + sp_stats['drawn']) / sp_stats['games']:.0%}"
                ),
                "decisive": f"{sp_stats['decisive'] / sp_stats['games']:.0%}",
            }

            if len(replay.rl_buf) > 0:
                losses = []

                n_pretrain = (
                    int(config.self_play_batch_size * config.self_play_pretrain_mix)
                    if len(replay.pretrain_buf) > 0
                    else 0
                )
                for _ in range(config.self_play_gradient_steps):
                    batch = replay.sample_rl(config.self_play_batch_size - n_pretrain)
                    if n_pretrain:
                        batch = batch + replay.sample_pretrain(
                            n_pretrain, require_policy=True
                        )
                    if batch:
                        losses.append(
                            train_batch(
                                train_model,
                                opt,
                                scaler,
                                batch,
                                device,
                                entropy_coef=config.self_play_entropy_coef,
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
                            **finish_postfix,
                            **elo_postfix,
                        }
                    )
                else:
                    pbar.set_postfix(
                        {"rl_buf": len(replay.rl_buf), **finish_postfix, **elo_postfix}
                    )
            else:
                pbar.set_postfix(
                    {"rl_buf": len(replay.rl_buf), **finish_postfix, **elo_postfix}
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        h2h = head_to_head_score(
            model,
            opponent_model,
            elo_state["best_state"],
            config.self_play_h2h_games * config.self_play_final_h2h_multiplier,
            config.self_play_max_moves,
            device,
            config,
            use_multiprocessing,
            max_workers=max_workers,
            executor=executor,
        )
        z = binomial_z_score(
            h2h["learner_wins"], h2h["drawn"], h2h["games"] - h2h["unresolved"]
        )
        record = f"{h2h['learner_wins']}-{h2h['opponent_wins']}-{h2h['drawn']}"
        pbar.write(f"[final] candidate scored {record} vs best (z={z:.2f})")
        if z > config.self_play_final_promote_z:
            elo_state["best_state"] = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            pbar.write(
                "[final] candidate beat the champion on the final match; keeping it"
            )
        else:
            model.load_state_dict(elo_state["best_state"])
            pbar.write("[final] restored the last promoted champion")
        final_elo, elo_state["elo_ema"] = estimate_elo(model, device, config, elo_state)
        pbar.write(f"[final] elo estimate: {final_elo:.0f}")


if __name__ == "__main__":
    from config import (
        Config,
        build_model,
        build_optimizer,
        build_scaler,
        build_scheduler,
        default_checkpoint_path,
        get_device,
        load_optimizer_state,
        optimizer_state_path,
        save_optimizer_state,
        set_optimizer_lr,
    )
    from model import save_checkpoint
    from dataset import DEFAULT_PATH, load_pretrain_dataset
    from replay_buffer import DualRingBuffer

    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()
    resuming = os.path.exists(checkpoint_path)
    print(
        f"Resuming self-play from {checkpoint_path}"
        if resuming
        else "Starting self-play from a randomly initialized model"
    )
    model, train_model = build_model(config, device, checkpoint_path)
    opt = set_optimizer_lr(build_optimizer(model, config), config.self_play_lr)
    scaler = build_scaler(device)
    total_steps = config.self_play_iterations * config.self_play_gradient_steps
    scheduler = build_scheduler(opt, total_steps)
    if resuming and os.path.exists(optimizer_state_path(checkpoint_path, "self_play")):
        load_optimizer_state(opt, scheduler, checkpoint_path, "self_play")
        if scheduler.last_epoch >= total_steps:
            opt.state.clear()
            set_optimizer_lr(opt, config.self_play_lr)
            scheduler = build_scheduler(opt, total_steps)
            print(
                "Prior LR schedule had already completed — starting a fresh "
                "schedule on top of the existing weights instead of resuming "
                "a spent one"
            )
        else:
            print("Resumed optimizer and LR schedule state from prior run")
    elif resuming:
        print(
            "No self-play optimizer state found; starting a fresh self-play "
            "optimizer and LR schedule from the checkpoint weights"
        )
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )

    if config.self_play_pretrain_mix > 0 and os.path.exists(DEFAULT_PATH):
        replay.extend_pretrain(
            load_pretrain_dataset(DEFAULT_PATH),
            pool_size=config.pretrain_shuffle_pool,
            chunk_size=config.pretrain_chunk_rows,
        )
        print(f"Anchoring RL batches with {len(replay.pretrain_buf):,} pretrain rows")
    elif config.self_play_pretrain_mix > 0:
        print(
            f"No pretrain dataset at {DEFAULT_PATH}; RL batches will be "
            "self-play samples only"
        )

    run_self_play(
        model, train_model, opt, scaler, scheduler, replay, device, config, {}
    )
    save_checkpoint(model, checkpoint_path)
    save_optimizer_state(opt, scheduler, checkpoint_path, "self_play")
