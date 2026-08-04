import copy
import gc
import itertools
import os
import random

import torch
from tqdm import tqdm

from evaluation import binomial_z_score, estimate_elo
from model import ChessNet
from self_play import generate_self_play_data, head_to_head_score, warmup_train_model
from training import train_batch


def clone_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def new_contender(state):
    return {"state": clone_state(state), "opt_state": None}


def load_contender(model, opt, contender):
    model.load_state_dict(contender["state"])
    if contender["opt_state"] is None:
        opt.state.clear()
    else:
        opt.load_state_dict(contender["opt_state"])


def save_contender(model, opt, contender):
    contender["state"] = clone_state(model.state_dict())
    contender["opt_state"] = copy.deepcopy(opt.state_dict())


def train_contender(
    model,
    train_model,
    opt,
    scaler,
    replay,
    device,
    config,
    opponent_model,
    opponent_states,
):
    losses, totals = [], {"games": 0, "decisive": 0, "drawn": 0, "unresolved": 0}
    for _ in range(config.population_generation_iters):
        opponent_state = (
            None
            if not opponent_states or random.random() < config.self_play_pool_self_prob
            else random.choice(opponent_states)
        )
        if opponent_state is not None:
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
            False,
            opponent_model=None if opponent_state is None else opponent_model,
            opponent_state_dict=opponent_state,
            value_smoothing=config.self_play_value_smoothing,
        )
        replay.extend_rl(samples)
        for key in totals:
            totals[key] += sp_stats[key]
        for _ in range(config.self_play_gradient_steps):
            batch = replay.sample_rl(config.self_play_batch_size)
            if batch:
                losses.append(train_batch(train_model, opt, scaler, batch, device))
    return losses, totals


def run_tournament(model, opponent_model, contenders, device, config):
    scores = [0.0] * len(contenders)
    for i, j in itertools.combinations(range(len(contenders)), 2):
        model.load_state_dict(contenders[i]["state"])
        stats = head_to_head_score(
            model,
            opponent_model,
            contenders[j]["state"],
            config.population_tournament_games,
            config.self_play_max_moves,
            device,
            config,
            False,
        )
        scores[i] += stats["learner_wins"] + 0.5 * stats["drawn"]
        scores[j] += stats["opponent_wins"] + 0.5 * stats["drawn"]
    return scores


def run_population_self_play(
    model, train_model, opt, scaler, replay, device, config, elo_state
):
    warmup_train_model(model, train_model, opt, scaler, config, device)
    contenders = [
        new_contender(model.state_dict()) for _ in range(config.population_size)
    ]
    anchor_state = clone_state(model.state_dict())

    opponent_model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
    ).to(device)
    opponent_model.eval()
    for p in opponent_model.parameters():
        p.requires_grad_(False)

    pbar = tqdm(range(config.population_generations), desc="Population Self-Play")
    last_elo_gen, ranking = 0, list(range(config.population_size))
    for gen in pbar:
        for idx, contender in enumerate(contenders):
            load_contender(model, opt, contender)
            opponent_states = [c["state"] for i, c in enumerate(contenders) if i != idx]
            losses, totals = train_contender(
                model,
                train_model,
                opt,
                scaler,
                replay,
                device,
                config,
                opponent_model,
                opponent_states,
            )
            save_contender(model, opt, contender)
            avg_loss = (
                sum(x[0] for x in losses) / len(losses) if losses else float("nan")
            )
            games = max(1, totals["games"])
            pbar.write(
                f"[gen {gen + 1}] contender {idx}: loss={avg_loss:.3f} "
                f"resolved={(totals['decisive'] + totals['drawn']) / games:.0%} "
                f"decisive={totals['decisive'] / games:.0%}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        scores = run_tournament(model, opponent_model, contenders, device, config)
        ranking = sorted(range(len(contenders)), key=lambda i: scores[i], reverse=True)
        pbar.write(
            f"[gen {gen + 1}] tournament: "
            + ", ".join(f"contender {i}={scores[i]:.1f}" for i in ranking)
        )

        survivors = ranking[: config.population_survivors]
        losers = ranking[config.population_survivors :]
        for slot, loser in enumerate(losers):
            contenders[loser] = new_contender(
                contenders[survivors[slot % len(survivors)]]["state"]
            )
        if losers:
            pbar.write(
                f"[gen {gen + 1}] kept {survivors}, replaced {losers} with clones of survivors"
            )

        if gen + 1 - last_elo_gen >= config.population_elo_refresh_generations:
            last_elo_gen = gen + 1
            model.load_state_dict(contenders[ranking[0]]["state"])
            _, elo_state["elo_ema"] = estimate_elo(model, device, config, elo_state)
            anchor_stats = head_to_head_score(
                model,
                opponent_model,
                anchor_state,
                config.self_play_h2h_games,
                config.self_play_max_moves,
                device,
                config,
                False,
            )
            anchor_z = binomial_z_score(
                anchor_stats["learner_wins"],
                anchor_stats["drawn"],
                anchor_stats["games"] - anchor_stats["unresolved"],
            )
            anchor_record = (
                f"{anchor_stats['learner_wins']}-"
                f"{anchor_stats['opponent_wins']}-{anchor_stats['drawn']}"
            )
            pbar.write(
                f"[gen {gen + 1}] leader elo_ema={elo_state['elo_ema']:.0f} vs pretrain anchor: "
                f"{anchor_record} (z={anchor_z:.2f})"
            )
            if anchor_z < -config.self_play_rollback_z:
                pbar.write(
                    f"[gen {gen + 1}] warning: population leader is losing to the original "
                    "pretrained checkpoint"
                )

    model.load_state_dict(contenders[ranking[0]]["state"])


if __name__ == "__main__":
    from config import (
        Config,
        build_model,
        build_optimizer,
        build_scaler,
        default_checkpoint_path,
        get_device,
        set_optimizer_lr,
    )
    from model import save_checkpoint
    from replay_buffer import DualRingBuffer

    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()
    print(
        f"Resuming population self-play from {checkpoint_path}"
        if os.path.exists(checkpoint_path)
        else "Starting population self-play from a randomly initialized model"
    )
    model, train_model = build_model(config, device, checkpoint_path)
    opt = set_optimizer_lr(build_optimizer(model, config), config.self_play_lr)
    scaler = build_scaler(device)
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )

    run_population_self_play(
        model, train_model, opt, scaler, replay, device, config, {}
    )
    save_checkpoint(model, checkpoint_path)
