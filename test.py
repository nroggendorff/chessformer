import argparse
import json
import math
import os
import random
from datetime import datetime, timezone

import chess
import chess.engine
from tqdm import tqdm

from config import Config, default_checkpoint_path, get_device
from evaluation import (
    clamp_uci_elo,
    fit_rating,
    opening_moves_for_game,
    play_eval_game,
    rating_standard_error,
)
from model import load_checkpoint
from policy import batched_policy_step

CHECKPOINT_PATH = default_checkpoint_path()

LADDER = [
    {"skill": 0, "depth": 1},
    {"skill": 0, "depth": 4},
    {"skill": 3, "depth": 6},
    {"skill": 6, "depth": 8},
    {"skill": 10, "depth": 10},
    {"skill": 14, "depth": 12},
    {"skill": 18, "depth": 13},
    {"skill": 20, "depth": 14},
    {"elo": 1320},
    {"elo": 1500},
    {"elo": 1700},
    {"elo": 1900},
    {"elo": 2100},
    {"elo": 2300},
    {"elo": 2400},
    {"elo": 2500},
    {"elo": 2600},
    {"elo": 2700},
    {"elo": 2900},
    {"elo": 3000},
]
START_INDEX = next(i for i, level in enumerate(LADDER) if "elo" in level)

GAMES_PER_LEVEL = 32
MAX_MOVES = 160
MOVETIME = 1.0
MCTS_SIMULATIONS = 800
ADJUDICATION_DEPTH = 14

MOVE_QUALITY_POSITIONS = 200
MOVE_QUALITY_DEPTH = 14
POSITION_SEED = 12345
MIN_PLY = 2
MAX_PLY = 40
MATE_SCORE = 1000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(SCRIPT_DIR, "eval_positions.fen")


def get_level_label(level):
    return (
        f"Elo {level['elo']}"
        if "elo" in level
        else f"skill {level['skill']}/depth {level['depth']}"
    )


def configure_engine(engine, level, movetime):
    if "elo" in level:
        elo = clamp_uci_elo(engine, level["elo"])
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        return chess.engine.Limit(time=movetime), {"elo": elo}

    engine.configure({"UCI_LimitStrength": False, "Skill Level": level["skill"]})
    return chess.engine.Limit(time=movetime, depth=level["depth"]), level


def play_level_games(
    stockfish_path,
    model,
    device,
    config,
    level,
    num_games,
    max_moves,
    movetime,
    mcts_simulations,
    adjudication_depth,
    opening_plies,
):
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        try:
            limit, actual_level = configure_engine(engine, level, movetime)
        except chess.engine.EngineError as e:
            print(f"Skipping {get_level_label(level)}: {e}")
            return None

        model.eval()
        label = get_level_label(actual_level)

        games = [
            play_eval_game(
                engine,
                model,
                device,
                config,
                i % 2 == 0,
                max_moves,
                limit,
                mcts_simulations=mcts_simulations,
                opening_moves=opening_moves_for_game(i, opening_plies),
                adjudication_depth=adjudication_depth,
            )
            for i in tqdm(range(num_games), desc=f"vs {label}", leave=False)
        ]

    scores = [g["score"] for g in games]
    return {
        "level": actual_level,
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


def run_adaptive_ladder(
    stockfish_path,
    model,
    device,
    config,
    games_per_level,
    max_moves,
    movetime,
    mcts_simulations,
    adjudication_depth,
    opening_plies,
):
    tested = {}

    for i in range(START_INDEX, len(LADDER)):
        result = play_level_games(
            stockfish_path,
            model,
            device,
            config,
            LADDER[i],
            games_per_level,
            max_moves,
            movetime,
            mcts_simulations,
            adjudication_depth,
            opening_plies,
        )
        if result:
            tested[i] = result
            if result["score"] == 0:
                break

    if START_INDEX in tested and tested[START_INDEX]["score"] == 0:
        for i in range(START_INDEX - 1, -1, -1):
            result = play_level_games(
                stockfish_path,
                model,
                device,
                config,
                LADDER[i],
                games_per_level,
                max_moves,
                movetime,
                mcts_simulations,
                adjudication_depth,
                opening_plies,
            )
            if result:
                tested[i] = result
                if result["score"] > 0:
                    break

    return [tested[i] for i in sorted(tested)]


def load_or_create_positions(path, num_positions, seed):
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    rng = random.Random(seed)
    positions = []

    for _ in range(num_positions):
        board = chess.Board()
        ply_count = rng.randint(MIN_PLY, MAX_PLY)

        for _ in range(ply_count):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))

        positions.append(board.fen())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(positions))

    return positions


def evaluate_move_quality(engine, model, device, positions, depth, multipv=8):
    samples = []
    for fen in tqdm(positions, desc="Move-quality analysis"):
        board = chess.Board(fen)
        if not list(board.legal_moves):
            continue

        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        infos = infos if isinstance(infos, list) else [infos]

        ranked_moves = [
            (info["pv"][0], info["score"].pov(board.turn).score(mate_score=MATE_SCORE))
            for info in infos
            if "pv" in info
        ]
        if not ranked_moves:
            continue

        best_sf_move, best_sf_score = ranked_moves[0]
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
            board.push(model_move)
            move_score = (
                engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
                .pov(not board.turn)
                .score(mate_score=MATE_SCORE)
            )
            board.pop()

        samples.append(
            {
                "centipawn_loss": best_sf_score - move_score,
                "rank": rank,
                "model_value": model_value,
                "stockfish_value": math.tanh(best_sf_score / 400.0),
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
        "value_mae": sum(abs(s["model_value"] - s["stockfish_value"]) for s in samples)
        / len(samples),
    }


def print_report(report):
    settings = report["settings"]
    print(
        "\nBenchmark budget: "
        f"{settings['mcts_simulations']} MCTS simulations, "
        f"{settings['movetime']:.3f}s Stockfish time, "
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
            f"\nEstimated Stockfish-calibrated rating: {report['estimated_rating']:.0f} "
            f"+/- {report['rating_stderr']:.0f}"
        )
        print(
            f"95% interval: {report['rating_ci95'][0]:.0f} to "
            f"{report['rating_ci95'][1]:.0f}"
        )
    else:
        print("\nNo calibrated Elo levels were reachable.")

    mq = report["move_quality"]
    if mq:
        print("\nMove quality vs Stockfish (depth-limited analysis):")
        for key, val in mq.items():
            if key != "positions":
                fmt = ".1%" if "rate" in key else (".3f" if "mae" in key else ".1f")
                print(f"  {key.replace('_', ' ')}: {val:{fmt}}")
        print(f"  positions analysed: {mq['positions']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--stockfish-path")
    parser.add_argument("--games", type=int, default=GAMES_PER_LEVEL)
    parser.add_argument("--max-moves", type=int, default=MAX_MOVES)
    parser.add_argument("--movetime", type=float, default=MOVETIME)
    parser.add_argument("--mcts-simulations", type=int, default=MCTS_SIMULATIONS)
    parser.add_argument("--adjudication-depth", type=int, default=ADJUDICATION_DEPTH)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument(
        "--move-quality-positions", type=int, default=MOVE_QUALITY_POSITIONS
    )
    parser.add_argument("--skip-move-quality", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        **({"stockfish_path": args.stockfish_path} if args.stockfish_path else {})
    )
    device = get_device()
    model = load_checkpoint(args.checkpoint, device, config)

    print(f"Loaded checkpoint: {args.checkpoint} on {device}")

    levels = run_adaptive_ladder(
        config.stockfish_path,
        model,
        device,
        config,
        args.games,
        args.max_moves,
        args.movetime,
        args.mcts_simulations,
        args.adjudication_depth,
        args.opening_plies,
    )
    calibrated = [lvl for lvl in levels if "elo" in lvl["level"]]

    rating = fit_rating(calibrated)
    rating_se = rating_standard_error(rating, calibrated)
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

        with chess.engine.SimpleEngine.popen_uci(config.stockfish_path) as engine:
            move_samples = evaluate_move_quality(
                engine, model, device, positions, MOVE_QUALITY_DEPTH
            )

        move_quality = summarize_move_quality(move_samples)

    report = {
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "estimated_rating": rating,
        "rating_stderr": rating_se,
        "rating_ci95": rating_ci95,
        "rating_reference": "Stockfish UCI_Elo calibration, not a human rating",
        "settings": {
            "games_per_level": args.games,
            "max_moves": args.max_moves,
            "movetime": args.movetime,
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
