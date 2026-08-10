import argparse
import json
import math
import os
import random
from datetime import datetime, timezone

from tqdm import tqdm

import pecan as pc
from config import Config, default_checkpoint_path, get_device
from evaluation import (
    fit_rating,
    play_eval_game,
    random_opening_moves,
    rating_for_depth,
    rating_standard_error,
)
from model import load_checkpoint
from oracle import Oracle
from policy import batched_policy_step

CHECKPOINT_PATH = default_checkpoint_path()

DEPTH_LADDER = [1, 2, 3, 4, 5, 6, 7, 8]

GAMES_PER_LEVEL = 32
MAX_MOVES = 100
MCTS_SIMULATIONS = 400
ADJUDICATION_DEPTH = 6

MOVE_QUALITY_POSITIONS = 200
MOVE_QUALITY_DEPTH = 6
POSITION_SEED = 12345
MIN_PLY = 2
MAX_PLY = 30
MATE_SCORE = 100000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(SCRIPT_DIR, "eval_positions.fen")


def play_level_games(
    oracle,
    model,
    device,
    config,
    depth,
    num_games,
    max_moves,
    mcts_simulations,
    adjudication_depth,
    opening_plies,
):
    label = f"oracle depth {depth}"
    games = [
        play_eval_game(
            oracle,
            model,
            device,
            config,
            i % 2 == 0,
            max_moves,
            depth,
            mcts_simulations=mcts_simulations,
            opening_moves=random_opening_moves(i, opening_plies),
            adjudication_depth=adjudication_depth,
        )
        for i in tqdm(range(num_games), desc=f"vs {label}", leave=False)
    ]
    scores = [g["score"] for g in games]
    return {
        "level": {"elo": rating_for_depth(depth), "depth": depth},
        "label": label,
        "games": num_games,
        "score": sum(scores),
        "wins": scores.count(1.0),
        "draws": scores.count(0.5),
        "losses": scores.count(0.0),
        "timeouts": sum(g["timed_out"] for g in games),
        "white_score": sum(g["score"] for g in games[0::2]),
        "black_score": sum(g["score"] for g in games[1::2]),
        "avg_plies": sum(g["plies"] for g in games) / num_games,
    }


def run_ladder(
    oracle,
    model,
    device,
    config,
    games_per_level,
    max_moves,
    mcts_simulations,
    adjudication_depth,
    opening_plies,
):
    results = []
    for depth in DEPTH_LADDER:
        result = play_level_games(
            oracle,
            model,
            device,
            config,
            depth,
            games_per_level,
            max_moves,
            mcts_simulations,
            adjudication_depth,
            opening_plies,
        )
        results.append(result)
        if result["score"] == 0:
            break
    return results


def load_or_create_positions(path, num_positions, seed):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    rng = random.Random(seed)
    positions = []

    for _ in range(num_positions):
        board = pc.Board()
        ply_count = rng.randint(MIN_PLY, MAX_PLY)

        for _ in range(ply_count):
            if board.is_game_over():
                break
            board.push(rng.choice(board.legal_moves))

        positions.append(board.fen())

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(positions))

    return positions


def evaluate_move_quality(oracle, model, device, positions, depth):
    samples = []
    for fen in tqdm(positions, desc="Move-quality analysis"):
        board = pc.Board(fen)
        if not board.legal_moves:
            continue

        result = oracle.search(board, depth=depth)
        ranked_moves = sorted(
            result["root_scores"], key=lambda pair: pair[1], reverse=True
        )
        if not ranked_moves:
            continue

        best_move, best_score = ranked_moves[0]
        moves, values, _, _ = batched_policy_step(
            [board], model, device, temperature=0.0
        )
        model_move, model_value = moves[0], values[0]

        rank, move_score = None, None
        for i, (mv, score) in enumerate(ranked_moves, 1):
            if mv == model_move:
                rank, move_score = i, score
                break

        if move_score is None:
            child = board.copy()
            child.push(model_move)
            move_score = -oracle.score_cp(child, depth=depth)

        samples.append(
            {
                "centipawn_loss": best_score - move_score,
                "rank": rank,
                "model_value": model_value,
                "oracle_value": math.tanh(best_score / 400.0),
            }
        )

    return samples


def summarize_move_quality(samples):
    if not samples:
        return {}
    losses = sorted(max(0, s["centipawn_loss"]) for s in samples)

    return {
        "positions": len(samples),
        "avg_centipawn_loss": sum(losses) / len(losses),
        "median_centipawn_loss": losses[len(losses) // 2],
        "top1_match_rate": sum(1 for s in samples if s["rank"] == 1) / len(samples),
        "top3_match_rate": sum(1 for s in samples if s["rank"] and s["rank"] <= 3)
        / len(samples),
        "value_mae": sum(abs(s["model_value"] - s["oracle_value"]) for s in samples)
        / len(samples),
    }


def print_report(report):
    settings = report["settings"]
    print(
        "\nBenchmark budget: "
        f"{settings['mcts_simulations']} MCTS simulations, "
        f"{settings['games_per_level']} games per level, "
        f"{settings['max_moves']} plies, "
        f"{settings['opening_plies']} opening plies"
    )
    print("\nLadder results (weakest to strongest):")
    for level in report["levels"]:
        print(
            f"  vs {level['label']}: {level['wins']}W {level['draws']}D {level['losses']}L "
            f"({level['score']:.1f}/{level['games']}), "
            f"white {level['white_score']:.1f}, black {level['black_score']:.1f}, "
            f"avg {level['avg_plies']:.0f} plies, {level['timeouts']} timeouts"
        )

    if report["estimated_rating"] is not None:
        print(
            f"\nEstimated rating (internal scale, oracle-calibrated, NOT a real chess rating): "
            f"{report['estimated_rating']:.0f} +/- {report['rating_stderr']:.0f}"
        )
        print(
            f"95% interval: {report['rating_ci95'][0]:.0f} to {report['rating_ci95'][1]:.0f}"
        )
    else:
        print("\nNo calibrated levels were reachable.")

    mq = report["move_quality"]
    if mq:
        print("\nMove quality vs oracle (depth-limited analysis):")
        for key, val in mq.items():
            if key != "positions":
                fmt = ".1%" if "rate" in key else (".3f" if "mae" in key else ".1f")
                print(f"  {key.replace('_', ' ')}: {val:{fmt}}")
        print(f"  positions analysed: {mq['positions']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--games", type=int, default=GAMES_PER_LEVEL)
    parser.add_argument("--max-moves", type=int, default=MAX_MOVES)
    parser.add_argument("--mcts-simulations", type=int, default=MCTS_SIMULATIONS)
    parser.add_argument("--adjudication-depth", type=int, default=ADJUDICATION_DEPTH)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument(
        "--move-quality-positions", type=int, default=MOVE_QUALITY_POSITIONS
    )
    parser.add_argument("--skip-move-quality", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    device = get_device()
    model = load_checkpoint(args.checkpoint, device, config)
    oracle = Oracle()

    print(f"Loaded checkpoint: {args.checkpoint} on {device}")

    levels = run_ladder(
        oracle,
        model,
        device,
        config,
        args.games,
        args.max_moves,
        args.mcts_simulations,
        args.adjudication_depth,
        args.opening_plies,
    )

    rating = fit_rating(levels)
    rating_se = rating_standard_error(rating, levels)
    rating_ci95 = (
        (rating - 1.96 * rating_se, rating + 1.96 * rating_se)
        if rating is not None and rating_se is not None
        else None
    )

    if args.skip_move_quality:
        move_quality = {}
    else:
        positions = load_or_create_positions(
            POSITIONS_FILE, args.move_quality_positions, POSITION_SEED
        )
        move_samples = evaluate_move_quality(
            oracle, model, device, positions, MOVE_QUALITY_DEPTH
        )
        move_quality = summarize_move_quality(move_samples)

    report = {
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "estimated_rating": rating,
        "rating_stderr": rating_se,
        "rating_ci95": rating_ci95,
        "rating_reference": "internal oracle-depth calibration, not a human or Stockfish rating",
        "settings": {
            "games_per_level": args.games,
            "max_moves": args.max_moves,
            "mcts_simulations": args.mcts_simulations,
            "adjudication_depth": args.adjudication_depth,
            "opening_plies": args.opening_plies,
        },
        "levels": levels,
        "move_quality": move_quality,
    }

    print_report(report)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
