import concurrent.futures
import itertools
import multiprocessing as mp
import os

from tqdm import tqdm

from evaluation import binomial_z_score, estimate_elo
from model import ChessNet, save_checkpoint
from population_workers import (
    calibrate_population_workers,
    worker_init,
    worker_train_contender,
)
from self_play import head_to_head_score
from state_utils import from_numpy_state, load_state, to_numpy_state


def clone_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def new_contender(state):
    return {"state": clone_state(state), "opt_state": None}


def run_tournament(model, opponent_model, contenders, device, config):
    scores = [0.0] * len(contenders)
    for i, j in itertools.combinations(range(len(contenders)), 2):
        load_state(model, contenders[i]["state"])
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


def run_population_self_play(model, device, config, elo_state, checkpoint_path=None):
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
    anchor_state_np = to_numpy_state(anchor_state)
    best_state, best_elo = None, None

    def make_pool():
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=worker_init,
            initargs=(device.type, config),
        )

    pool = make_pool()
    try:
        for gen in pbar:
            while True:
                try:
                    futures = {
                        pool.submit(
                            worker_train_contender,
                            to_numpy_state(contender["state"]),
                            to_numpy_state(contender["opt_state"]),
                            [
                                to_numpy_state(c["state"])
                                for i, c in enumerate(contenders)
                                if i != idx
                            ],
                            anchor_state_np,
                            config,
                        ): idx
                        for idx, contender in enumerate(contenders)
                    }
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        state, opt_state, losses, totals = future.result()
                        contenders[idx]["state"] = from_numpy_state(state)
                        contenders[idx]["opt_state"] = from_numpy_state(opt_state)
                        avg_loss = (
                            sum(x[0] for x in losses) / len(losses)
                            if losses
                            else float("nan")
                        )
                        games = max(1, totals["games"])
                        pbar.write(
                            f"[gen {gen + 1}] contender {idx}: loss={avg_loss:.3f} "
                            f"resolved={(totals['decisive'] + totals['drawn']) / games:.0%} "
                            f"decisive={totals['decisive'] / games:.0%}"
                        )
                    break
                except concurrent.futures.process.BrokenProcessPool:
                    pbar.write(
                        f"[gen {gen + 1}] worker pool crashed, restarting workers and retrying"
                    )
                    pool.shutdown(wait=False)
                    pool = make_pool()

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
                load_state(model, contenders[ranking[0]]["state"])
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

                if best_elo is None or elo_state["elo_ema"] > best_elo:
                    best_elo = elo_state["elo_ema"]
                    best_state = clone_state(contenders[ranking[0]]["state"])
                    pbar.write(f"[gen {gen + 1}] new best: elo_ema={best_elo:.0f}")
                    if checkpoint_path is not None:
                        load_state(model, best_state)
                        save_checkpoint(model, checkpoint_path)
                        pbar.write(
                            f"[gen {gen + 1}] checkpoint saved to {checkpoint_path}"
                        )
                elif (
                    elo_state["elo_ema"] < best_elo - config.population_rollback_margin
                ):
                    pbar.write(
                        f"[gen {gen + 1}] rolling back population: elo_ema dropped to "
                        f"{elo_state['elo_ema']:.0f} from best {best_elo:.0f}"
                    )
                    contenders = [
                        new_contender(best_state) for _ in range(config.population_size)
                    ]
                    ranking = list(range(config.population_size))
                    elo_state["elo_ema"] = best_elo
    finally:
        pool.shutdown(wait=False)

    load_state(
        model, best_state if best_state is not None else contenders[ranking[0]]["state"]
    )


if __name__ == "__main__":
    from config import Config, build_model, default_checkpoint_path, get_device

    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()
    print(
        f"Resuming population self-play from {checkpoint_path}"
        if os.path.exists(checkpoint_path)
        else "Starting population self-play from a randomly initialized model"
    )
    model, _ = build_model(config, device, checkpoint_path, compile_model=False)

    run_population_self_play(model, device, config, {}, checkpoint_path=checkpoint_path)
    save_checkpoint(model, checkpoint_path)
