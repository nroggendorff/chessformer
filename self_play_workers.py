import concurrent.futures
import gc
import multiprocessing as mp
import random
import resource

import chess.engine
import numpy as np
import torch

from config import amp_dtype
from evaluation import configure_anchor
from model import ChessNet
from self_play_game import play_games_batched

_GLOBAL_MODEL = None
_GLOBAL_OPPONENT = None
_GLOBAL_STOCKFISH = None
_GLOBAL_STOCKFISH_ANCHOR = None
_GLOBAL_STOCKFISH_LIMIT = None
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


def ensure_stockfish(path, anchor):
    global _GLOBAL_STOCKFISH, _GLOBAL_STOCKFISH_ANCHOR, _GLOBAL_STOCKFISH_LIMIT
    if anchor["engine_prob"] <= 0.0:
        return None, None
    if _GLOBAL_STOCKFISH is None:
        _GLOBAL_STOCKFISH = chess.engine.SimpleEngine.popen_uci(path)
    if anchor != _GLOBAL_STOCKFISH_ANCHOR:
        _GLOBAL_STOCKFISH_LIMIT = configure_anchor(_GLOBAL_STOCKFISH, anchor)
        _GLOBAL_STOCKFISH_ANCHOR = anchor
    return _GLOBAL_STOCKFISH, _GLOBAL_STOCKFISH_LIMIT


def _proc_status_mb(field):
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def anonymous_rss_mb():
    value = _proc_status_mb("RssAnon")
    if value is None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return value


def resident_rss_mb():
    value = _proc_status_mb("VmRSS")
    if value is None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return value


def worker_probe_footprint(
    num_games,
    max_moves,
    mcts_simulations,
    sims_per_wave,
    target_batch_size,
    max_batch_size,
    c_puct,
    fpu_reduction,
    device_type,
):
    device = torch.device(device_type)
    with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
        play_games_batched(
            _GLOBAL_MODEL,
            device,
            num_games=num_games,
            max_moves=max_moves,
            sample_moves=max_moves,
            mcts_simulations=mcts_simulations,
            opponent_mcts_simulations=mcts_simulations,
            sims_per_wave=sims_per_wave,
            target_batch_size=target_batch_size,
            max_batch_size=max_batch_size,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            record_trajectory=False,
        )
    return anonymous_rss_mb()


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
        worker_rss_mb = probe.submit(
            worker_probe_footprint,
            min(config.self_play_chunk_games, config.self_play_games_per_iter),
            config.self_play_probe_max_moves,
            config.self_play_mcts_simulations,
            config.mcts_sims_per_wave,
            config.mcts_target_batch_size,
            config.mcts_max_batch_size,
            config.mcts_c_puct,
            config.mcts_fpu_reduction,
            device.type,
        ).result()

    from config import cgroup_memory_limit_mb

    vram_workers = worker_vram_ceiling(device, config)
    memory_limit_mb = cgroup_memory_limit_mb()
    if memory_limit_mb is None:
        print(f"Self-play worker baseline: {worker_rss_mb:.0f} MB/process")
        return min(config.self_play_max_workers, vram_workers)

    main_rss_mb = resident_rss_mb()
    budgeted_worker_mb = worker_rss_mb * config.self_play_worker_rss_headroom
    budget_mb = memory_limit_mb - main_rss_mb - config.self_play_memory_safety_margin_mb
    safe_workers = max(1, int(budget_mb // budgeted_worker_mb))
    workers = min(config.self_play_max_workers, safe_workers, vram_workers)
    print(
        f"Self-play workers: {workers} "
        f"(~{worker_rss_mb:.0f} MB/worker measured, {budgeted_worker_mb:.0f} MB budgeted; "
        f"main {main_rss_mb:.0f} MB, limit {memory_limit_mb:.0f} MB)"
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
    stockfish_anchor=None,
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
    material_scale=0.0,
    material_value_weight=0.5,
    draw_material_weight=0.0,
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

    stockfish_engine, stockfish_limit = None, None
    if stockfish_anchor is not None:
        try:
            stockfish_engine, stockfish_limit = ensure_stockfish(
                stockfish_path, stockfish_anchor
            )
        except (FileNotFoundError, chess.engine.EngineError) as error:
            print(f"Stockfish opponent unavailable in worker: {error}")
            stockfish_anchor = None

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
            stockfish_anchor=stockfish_anchor,
            stockfish_limit=stockfish_limit,
            stockfish_movetime=stockfish_movetime,
            material_scale=material_scale,
            material_value_weight=material_value_weight,
            draw_material_weight=draw_material_weight,
            resign_threshold=resign_threshold,
            resign_streak=resign_streak,
            add_root_noise=add_root_noise,
            value_smoothing=value_smoothing,
            record_trajectory=record_trajectory,
            include_policy_q_threshold=include_policy_q_threshold,
            opening_moves_per_game=opening_moves_per_game,
            target_beta=target_beta,
        )
