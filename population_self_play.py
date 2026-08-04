import concurrent.futures
import itertools
import multiprocessing as mp
import os

from tqdm import tqdm

from evaluation import binomial_z_score, estimate_elo
from model import ChessNet
from population_workers import (
    calibrate_population_workers,
    worker_init,
    worker_train_contender,
)
from self_play import head_to_head_score


def clone_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def new_contender(state):
    return {"state": clone_state(state), "opt_state": None}


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


def run_population_self_play(model, device, config, elo_state):
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
    num_workers = calibrate_population_workers(device, config)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
        initializer=worker_init,
        initargs=(device.type, config),
    ) as pool:
        for gen in pbar:
            futures = {
                pool.submit(
                    worker_train_contender,
                    contender["state"],
                    contender["opt_state"],
                    [c["state"] for i, c in enumerate(contenders) if i != idx],
                    config,
                ): idx
                for idx, contender in enumerate(contenders)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                state, opt_state, losses, totals = future.result()
                contenders[idx]["state"] = state
                contenders[idx]["opt_state"] = opt_state
                avg_loss = (
                    sum(x[0] for x in losses) / len(losses) if losses else float("nan")
                )
                games = max(1, totals["games"])
                pbar.write(
                    f"[gen {gen + 1}] contender {idx}: loss={avg_loss:.3f} "
                    f"resolved={(totals['decisive'] + totals['drawn']) / games:.0%} "
                    f"decisive={totals['decisive'] / games:.0%}"
                )

            scores = run_tournament(model, opponent_model, contenders, device, config)
            ranking = sorted(
                range(len(contenders)), key=lambda i: scores[i], reverse=True
            )
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
                        f"[gen {gen + 1}] warning: population leader is losing to the "
                        "original pretrained checkpoint"
                    )

    model.load_state_dict(contenders[ranking[0]]["state"])


if __name__ == "__main__":
    from config import Config, build_model, default_checkpoint_path, get_device
    from model import save_checkpoint

    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()
    print(
        f"Resuming population self-play from {checkpoint_path}"
        if os.path.exists(checkpoint_path)
        else "Starting population self-play from a randomly initialized model"
    )
    model, _ = build_model(config, device, checkpoint_path, compile_model=False)

    run_population_self_play(model, device, config, {})
    save_checkpoint(model, checkpoint_path)
