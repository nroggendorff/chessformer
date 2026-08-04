import copy
import gc
import random

import torch

from config import build_model, build_optimizer, build_scaler, set_optimizer_lr
from model import ChessNet
from replay_buffer import DualRingBuffer
from self_play import generate_self_play_data, warmup_train_model
from training import train_batch

_MODEL = _TRAIN_MODEL = _OPT = _SCALER = _OPPONENT = _REPLAY = _DEVICE = None


def worker_init(device_type, config):
    global _MODEL, _TRAIN_MODEL, _OPT, _SCALER, _OPPONENT, _REPLAY, _DEVICE
    _DEVICE = torch.device(device_type)
    _MODEL, _TRAIN_MODEL = build_model(config, _DEVICE)
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


def worker_train_contender(state, opt_state, opponent_states, config):
    _MODEL.load_state_dict(state)
    if opt_state is None:
        _OPT.state.clear()
    else:
        _OPT.load_state_dict(opt_state)

    losses, totals = [], {"games": 0, "decisive": 0, "drawn": 0, "unresolved": 0}
    for _ in range(config.population_generation_iters):
        opponent_state = (
            None
            if not opponent_states or random.random() < config.self_play_pool_self_prob
            else random.choice(opponent_states)
        )
        if opponent_state is not None:
            _OPPONENT.load_state_dict(opponent_state)
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
        {k: v.cpu().clone() for k, v in _MODEL.state_dict().items()},
        copy.deepcopy(_OPT.state_dict()),
        losses,
        totals,
    )
