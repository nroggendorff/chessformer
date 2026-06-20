import concurrent.futures
import contextlib
import gc
import multiprocessing as mp
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from mcts import play_games_batched
from model import ChessNet
from training import train_batch

_GLOBAL_MODEL = None


def worker_init(device_type):
    global _GLOBAL_MODEL
    torch.set_num_threads(1)
    _GLOBAL_MODEL = ChessNet().to(torch.device(device_type))


def worker_play_games(
    state_dict, seed, num_games, sims, max_moves, sample_moves, device_type
):
    global _GLOBAL_MODEL
    assert _GLOBAL_MODEL
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_type)
    _GLOBAL_MODEL.load_state_dict(state_dict)
    _GLOBAL_MODEL.eval()

    with torch.inference_mode():
        return play_games_batched(
            _GLOBAL_MODEL,
            device,
            num_games=num_games,
            sims=sims,
            max_moves=max_moves,
            sample_moves=sample_moves,
        )


def generate_self_play_data(
    model,
    total_games,
    sims,
    max_moves,
    sample_moves,
    device,
    max_workers=None,
    executor=None,
):
    if device.type == "cuda":
        with torch.autocast(
            device_type="cuda", dtype=torch.float16
        ), torch.inference_mode():
            return play_games_batched(
                model,
                device,
                num_games=total_games,
                sims=sims,
                max_moves=max_moves,
                sample_moves=sample_moves,
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
                sims,
                max_moves,
                sample_moves,
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
        initargs=(device.type,),
    ) as fresh_executor:
        futures = submit(fresh_executor)
        return [s for f in concurrent.futures.as_completed(futures) for s in f.result()]


def run_self_play(
    model,
    opt,
    scaler,
    replay,
    device,
    iterations=100,
    games_per_iter=128,
    mcts_sims=100,
    max_moves=120,
    sample_moves=15,
    batch_size=2048,
    gradient_steps=15,
    self_play_workers=None,
):
    use_multiprocessing = device.type != "cuda"
    max_workers = (
        min(self_play_workers or mp.cpu_count(), games_per_iter)
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
            initargs=(device.type,),
        )
        if use_multiprocessing
        else contextlib.nullcontext()
    )

    with executor_cm as executor:
        pbar = tqdm(range(iterations), desc="Self-Play RL Optimization")
        for _ in pbar:
            replay.extend_rl(
                generate_self_play_data(
                    model,
                    games_per_iter,
                    mcts_sims,
                    max_moves,
                    sample_moves,
                    device,
                    max_workers=max_workers,
                    executor=executor,
                )
            )

            if len(replay.pretrain_buf) > 0 or len(replay.rl_buf) > 0:
                losses = [
                    train_batch(
                        model,
                        opt,
                        scaler,
                        replay.sample(batch_size, mix_ratio=0.5),
                        device,
                    )
                    for _ in range(gradient_steps)
                ]
                avg_loss, avg_p, avg_v = (
                    sum(x[i] for x in losses) / len(losses) for i in range(3)
                )
                pbar.set_postfix(
                    loss=f"{avg_loss:.3f}",
                    policy=f"{avg_p:.3f}",
                    value=f"{avg_v:.3f}",
                    rl_buf=len(replay.rl_buf),
                )
            else:
                pbar.set_postfix(rl_buf=len(replay.rl_buf))

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
