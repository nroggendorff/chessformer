import concurrent.futures
import gc
import multiprocessing as mp
import random
import resource

import chess.engine
import numpy as np
import torch

from config import amp_dtype
from evaluation import clamp_uci_elo
from model import ChessNet
from self_play_game import play_games_batched

_GLOBAL_MODEL = None
_GLOBAL_OPPONENT = None
_GLOBAL_STOCKFISH = None
_WORKER_MODEL_ARGS = None


def worker_init(device_type, d_model, nhead, enc_layers, heatmap_hidden):
    global _GLOBAL_MODEL, _WORKER_MODEL_ARGS
    gc.set_threshold(100000, 50, 50)
    torch.set_num_threads(1)
    _WORKER_MODEL_ARGS = (device_type, d_model, nhead, enc_layers, heatmap_hidden)
    _GLOBAL_MODEL = ChessNet(
        d_model=d_model,
        nhead=nhead,
        enc_layers=enc_layers,
        heatmap_hidden=heatmap_hidden,
    ).to(torch.device(device_type))


def ensure_opponent():
    global _GLOBAL_OPPONENT
    if _GLOBAL_OPPONENT is None:
        device_type, d_model, nhead, enc_layers, heatmap_hidden = _WORKER_MODEL_ARGS
        _GLOBAL_OPPONENT = ChessNet(
            d_model=d_model,
            nhead=nhead,
            enc_layers=enc_layers,
            heatmap_hidden=heatmap_hidden,
        ).to(torch.device(device_type))
    return _GLOBAL_OPPONENT


def ensure_stockfish(path, uci_elo):
    global _GLOBAL_STOCKFISH
    if _GLOBAL_STOCKFISH is None:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": clamp_uci_elo(engine, uci_elo)}
        )
        _GLOBAL_STOCKFISH = engine
    return _GLOBAL_STOCKFISH


def worker_report_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def worker_vram_ceiling(device, config):
    if device.type != "cuda":
        return config.self_play_max_workers
    free_bytes, _ = torch.cuda.mem_get_info()
    free_mb = free_bytes / (1024**2)
    budget_mb = free_mb - config.self_play_memory_safety_margin_mb
    return max(1, int(budget_mb // config.self_play_worker_vram_mb))


def calibrate_self_play_workers(config, device):
    if config.self_play_max_workers <= 1:
        return config.self_play_max_workers

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=worker_init,
        initargs=(
            device.type,
            config.d_model,
            config.nhead,
            config.enc_layers,
            config.heatmap_hidden,
        ),
    ) as probe:
        worker_rss_mb = probe.submit(worker_report_rss_mb).result()

    from config import cgroup_memory_limit_mb

    vram_workers = worker_vram_ceiling(device, config)
    memory_limit_mb = cgroup_memory_limit_mb()
    if memory_limit_mb is None:
        print(f"Self-play worker baseline: {worker_rss_mb:.0f} MB/process")
        return min(config.self_play_max_workers, vram_workers)

    main_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    budget_mb = memory_limit_mb - main_rss_mb - config.self_play_memory_safety_margin_mb
    safe_workers = max(1, int(budget_mb // worker_rss_mb))
    workers = min(config.self_play_max_workers, safe_workers, vram_workers)
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
    timeout_value_weight,
    mcts_simulations,
    opponent_mcts_simulations,
    sims_per_wave,
    target_batch_size,
    max_batch_size,
    c_puct,
    dirichlet_alpha,
    root_noise_frac,
    device_type,
    opponent_state_dict=None,
    stockfish_path=None,
    stockfish_elo=1800,
    stockfish_movetime=0.1,
    resign_threshold=None,
    resign_streak=2,
    add_root_noise=True,
    value_smoothing=0.0,
    record_trajectory=True,
    include_policy_q_threshold=0.85,
    opening_moves_per_game=None,
    target_beta=0.0,
    fpu_reduction=0.25,
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

    stockfish_engine = None
    if stockfish_path is not None:
        try:
            stockfish_engine = ensure_stockfish(stockfish_path, stockfish_elo)
        except (FileNotFoundError, chess.engine.EngineError) as error:
            print(f"Stockfish opponent unavailable in worker: {error}")

    with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
        return play_games_batched(
            _GLOBAL_MODEL,
            device,
            num_games=num_games,
            max_moves=max_moves,
            sample_moves=sample_moves,
            temperature=temperature,
            temperature_floor=temperature_floor,
            decisive_weight=decisive_weight,
            timeout_value_weight=timeout_value_weight,
            mcts_simulations=mcts_simulations,
            opponent_mcts_simulations=opponent_mcts_simulations,
            sims_per_wave=sims_per_wave,
            target_batch_size=target_batch_size,
            max_batch_size=max_batch_size,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            dirichlet_alpha=dirichlet_alpha,
            root_noise_frac=root_noise_frac,
            opponent_model=opponent,
            stockfish_engine=stockfish_engine,
            stockfish_movetime=stockfish_movetime,
            resign_threshold=resign_threshold,
            resign_streak=resign_streak,
            add_root_noise=add_root_noise,
            value_smoothing=value_smoothing,
            record_trajectory=record_trajectory,
            include_policy_q_threshold=include_policy_q_threshold,
            opening_moves_per_game=opening_moves_per_game,
            target_beta=target_beta,
        )
