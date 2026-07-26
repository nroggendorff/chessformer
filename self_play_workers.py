import concurrent.futures
import gc
import multiprocessing as mp
import random
import resource

import numpy as np
import torch

from model import ChessNet
from self_play_game import play_games_batched

_GLOBAL_MODEL = None
_GLOBAL_OPPONENT = None
_WORKER_MODEL_ARGS = None


def worker_init(device_type, d_model, nhead, enc_layers, value_hidden, attn_rank):
    global _GLOBAL_MODEL, _WORKER_MODEL_ARGS
    gc.set_threshold(100000, 50, 50)
    torch.set_num_threads(1)
    _WORKER_MODEL_ARGS = (
        device_type,
        d_model,
        nhead,
        enc_layers,
        value_hidden,
        attn_rank,
    )
    _GLOBAL_MODEL = ChessNet(
        d_model=d_model,
        nhead=nhead,
        enc_layers=enc_layers,
        value_hidden=value_hidden,
        attn_rank=attn_rank,
    ).to(torch.device(device_type))


def ensure_opponent():
    global _GLOBAL_OPPONENT
    if _GLOBAL_OPPONENT is None:
        device_type, d_model, nhead, enc_layers, value_hidden, attn_rank = (
            _WORKER_MODEL_ARGS
        )
        _GLOBAL_OPPONENT = ChessNet(
            d_model=d_model,
            nhead=nhead,
            enc_layers=enc_layers,
            value_hidden=value_hidden,
            attn_rank=attn_rank,
        ).to(torch.device(device_type))
    return _GLOBAL_OPPONENT


def worker_report_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def calibrate_self_play_workers(config):
    if config.self_play_max_workers <= 1:
        return config.self_play_max_workers

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=worker_init,
        initargs=(
            "cpu",
            config.d_model,
            config.nhead,
            config.enc_layers,
            config.value_hidden,
            config.attn_type_rank,
        ),
    ) as probe:
        worker_rss_mb = probe.submit(worker_report_rss_mb).result()

    from config import cgroup_memory_limit_mb

    memory_limit_mb = cgroup_memory_limit_mb()
    if memory_limit_mb is None:
        print(f"Self-play worker baseline: {worker_rss_mb:.0f} MB/process")
        return config.self_play_max_workers

    main_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    budget_mb = memory_limit_mb - main_rss_mb - config.self_play_memory_safety_margin_mb
    safe_workers = max(1, int(budget_mb // worker_rss_mb))
    workers = min(config.self_play_max_workers, safe_workers)
    print(
        f"Self-play workers: {workers} "
        f"(~{worker_rss_mb:.0f} MB/worker, {memory_limit_mb:.0f} MB available)"
    )
    return workers


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
    global _GLOBAL_MODEL
    assert _GLOBAL_MODEL
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_type)
    _GLOBAL_MODEL.load_state_dict(state_dict)
    _GLOBAL_MODEL.eval()
    opponent = None
    if opponent_state_dict is not None:
        opponent = ensure_opponent()
        opponent.load_state_dict(opponent_state_dict)
        opponent.eval()

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
