import concurrent.futures
import gc
import multiprocessing as mp
import random

import torch

from config import build_model, build_optimizer, build_scaler, set_optimizer_lr
from model import ChessNet
from replay_buffer import DualRingBuffer
from self_play import generate_self_play_data, warmup_train_model
from state_utils import from_numpy_state, load_state, to_numpy_state
from training import train_batch

_MODEL = _TRAIN_MODEL = _OPT = _SCALER = _OPPONENT = _REPLAY = _DEVICE = None


def worker_init(device_type, config):
    global _MODEL, _TRAIN_MODEL, _OPT, _SCALER, _OPPONENT, _REPLAY, _DEVICE
    _DEVICE = torch.device(device_type)
    _MODEL, _TRAIN_MODEL = build_model(config, _DEVICE, compile_model=False)
    _OPT = set_optimizer_lr(build_optimizer(_MODEL, config), config.self_play_lr)
    _SCALER = build_scaler(_DEVICE)
    _OPPONENT = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
    ).to(_DEVICE)
    _OPPONENT.eval()
    for p in _OPPONENT.parameters():
        p.requires_grad_(False)
    _REPLAY = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )
    warmup_train_model(_MODEL, _TRAIN_MODEL, _OPT, _SCALER, config, _DEVICE)


def worker_report_cuda_mb():
    return torch.cuda.memory_reserved(_DEVICE) / (1024**2)


def calibrate_population_workers(device, config):
    if device.type != "cuda" or config.population_size <= 1:
        return config.population_size
    free_bytes, _ = torch.cuda.mem_get_info(device.index or 0)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=worker_init,
        initargs=(device.type, config),
    ) as probe:
        worker_mb = probe.submit(worker_report_cuda_mb).result()
    budget_mb = free_bytes / (1024**2) - config.population_memory_safety_margin_mb
    workers = max(1, min(config.population_size, int(budget_mb // worker_mb)))
    print(
        f"Population self-play workers: {workers} "
        f"(~{worker_mb:.0f} MB/worker, {budget_mb:.0f} MB budget)"
    )
    return workers


def worker_train_contender(state, opt_state, opponent_states, anchor_state, config):
    load_state(_MODEL, state)
    if opt_state is None:
        _OPT.state.clear()
    else:
        _OPT.load_state_dict(from_numpy_state(opt_state))

    losses, totals = [], {"games": 0, "decisive": 0, "drawn": 0, "unresolved": 0}
    for _ in range(config.population_generation_iters):
        roll = random.random()
        if roll < config.self_play_pool_self_prob:
            opponent_state = None
        elif roll < config.self_play_pool_self_prob + config.self_play_anchor_prob:
            opponent_state = from_numpy_state(anchor_state)
        elif opponent_states:
            opponent_state = from_numpy_state(random.choice(opponent_states))
        else:
            opponent_state = None
        if opponent_state is not None:
            load_state(_OPPONENT, opponent_state)
        samples, sp_stats = generate_self_play_data(
            _MODEL,
            config.self_play_games_per_iter,
            config.self_play_max_moves,
            config.self_play_sample_moves,
            config.self_play_temperature,
            config.self_play_temperature_floor,
            _DEVICE,
            config,
            False,
            opponent_model=None if opponent_state is None else _OPPONENT,
            opponent_state_dict=opponent_state,
            value_smoothing=config.self_play_value_smoothing,
        )
        _REPLAY.extend_rl(samples)
        for key in totals:
            totals[key] += sp_stats[key]
        for _ in range(config.self_play_gradient_steps):
            batch = _REPLAY.sample_rl(config.self_play_batch_size)
            if batch:
                losses.append(train_batch(_TRAIN_MODEL, _OPT, _SCALER, batch, _DEVICE))

    if _DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return (
        to_numpy_state(_MODEL.state_dict()),
        to_numpy_state(_OPT.state_dict()),
        losses,
        totals,
    )
