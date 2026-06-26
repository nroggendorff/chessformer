import json
import math
import os
import random
from datetime import datetime, timezone

import chess
import chess.engine
from tqdm import tqdm

from config import Config, get_device
from evaluation import clamp_uci_elo, play_eval_game
from model import load_checkpoint
from policy import batched_policy_step

CHECKPOINT_PATH = "/opt/ml/model/chessformer.safetensors"

WEAK_LADDER = [
    {"skill": 0, "depth": 1},
    {"skill": 0, "depth": 4},
    {"skill": 3, "depth": 6},
    {"skill": 6, "depth": 8},
    {"skill": 10, "depth": 10},
    {"skill": 14, "depth": 12},
    {"skill": 18, "depth": 13},
    {"skill": 20, "depth": 14},
]
CALIBRATED_LADDER = [
    {"elo": 1320},
    {"elo": 1500},
    {"elo": 1700},
    {"elo": 1900},
    {"elo": 2100},
    {"elo": 2400},
    {"elo": 2700},
    {"elo": 3000},
]
LADDER = WEAK_LADDER + CALIBRATED_LADDER
START_INDEX = len(WEAK_LADDER)

GAMES_PER_LEVEL = 8
MAX_MOVES = 80
MOVETIME = 0.2

MOVE_QUALITY_POSITIONS = 200
MOVE_QUALITY_DEPTH = 14
POSITION_SEED = 12345
MIN_PLY = 2
MAX_PLY = 40

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(SCRIPT_DIR, "eval_positions.fen")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "reports")


def level_label(level):
    return (
        f"Elo {level['elo']}"
        if "elo" in level
        else f"skill {level['skill']}/depth {level['depth']}"
    )


def configure_level(engine, level):
    if "elo" in level:
        elo = clamp_uci_elo(engine, level["elo"])
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        return chess.engine.Limit(time=MOVETIME), {"elo": elo}
    engine.configure({"UCI_LimitStrength": False, "Skill Level": level["skill"]})
    return chess.engine.Limit(time=MOVETIME, depth=level["depth"]), level


def play_level_games(stockfish_path, model, device, level, num_games, max_moves):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        limit, actual_level = configure_level(engine, level)
    except chess.engine.EngineError as e:
        engine.quit()
        print(f"Skipping {level_label(level)}: {e}")
        return None

    model.eval()
    label = level_label(actual_level)
    games = [
        play_eval_game(engine, model, device, i % 2 == 0, max_moves, limit)
        for i in tqdm(range(num_games), desc=f"vs {label}", leave=False)
    ]
    engine.quit()

    white_games, black_games = games[0::2], games[1::2]
    return {
        "level": actual_level,
        "label": label,
        "games": num_games,
        "score": sum(g["score"] for g in games),
        "wins": sum(1 for g in games if g["score"] == 1.0),
        "draws": sum(1 for g in games if g["score"] == 0.5),
        "losses": sum(1 for g in games if g["score"] == 0.0),
        "white_score": sum(g["score"] for g in white_games),
        "black_score": sum(g["score"] for g in black_games),
        "avg_plies": sum(g["plies"] for g in games) / num_games,
    }


def run_adaptive_ladder(stockfish_path, model, device):
    tested = {}

    idx = START_INDEX
    while idx >= 0:
        result = play_level_games(
            stockfish_path, model, device, LADDER[idx], GAMES_PER_LEVEL, MAX_MOVES
        )
        if result is not None:
            tested[idx] = result
            if result["score"] > 0:
                break
        idx -= 1

    idx = START_INDEX + 1
    while idx < len(LADDER):
        result = play_level_games(
            stockfish_path, model, device, LADDER[idx], GAMES_PER_LEVEL, MAX_MOVES
        )
        if result is not None:
            tested[idx] = result
            if result["score"] == 0:
                break
        idx += 1

    return [tested[i] for i in sorted(tested)]


def expected_score(rating, anchor):
    return 1 / (1 + 10 ** ((anchor - rating) / 400))


def fit_rating(calibrated_results, lo=-3000.0, hi=4000.0, iters=80):
    total_actual = sum(r["score"] for r in calibrated_results)
    for _ in range(iters):
        mid = (lo + hi) / 2
        total_expected = sum(
            r["games"] * expected_score(mid, r["level"]["elo"])
            for r in calibrated_results
        )
        lo, hi = (mid, hi) if total_expected < total_actual else (lo, mid)
    return (lo + hi) / 2


def rating_standard_error(rating, calibrated_results):
    information = sum(
        r["games"]
        * expected_score(rating, r["level"]["elo"])
        * (1 - expected_score(rating, r["level"]["elo"]))
        for r in calibrated_results
    )
    return (
        400 / (math.log(10) * math.sqrt(information))
        if information > 0
        else float("inf")
    )


def load_or_create_positions(path, num_positions, seed, min_ply, max_ply):
    if os.path.exists(path):
        return [line.strip() for line in open(path) if line.strip()]

    rng = random.Random(seed)
    positions = []
    for _ in range(num_positions):
        board = chess.Board()
        for _ in range(rng.randint(min_ply, max_ply)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        positions.append(board.fen())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(positions))
    return positions


def evaluate_move_quality(engine, model, device, positions, depth, multipv=8):
    samples = []
    for fen in tqdm(positions, desc="Move-quality analysis"):
        board = chess.Board(fen)
        if not list(board.legal_moves):
            continue

        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        if not isinstance(infos, list):
            infos = [infos]
        ranked = [
            (info["pv"][0], info["score"].pov(board.turn).score(mate_score=10000))
            for info in infos
            if "pv" in info
        ]
        if not ranked:
            continue

        moves, values, _ = batched_policy_step([board], model, device, temperature=0.0)
        move, model_value, mover = moves[0], values[0], board.turn

        match = next(
            ((rank, score) for rank, (mv, score) in enumerate(ranked, 1) if mv == move),
            None,
        )
        if match is not None:
            rank, move_score = match
        else:
            board.push(move)
            move_score = (
                engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
                .pov(mover)
                .score(mate_score=10000)
            )
            board.pop()
            rank = None

        samples.append(
            {
                "centipawn_loss": ranked[0][1] - move_score,
                "rank": rank,
                "model_value": model_value,
                "stockfish_value": math.tanh(ranked[0][1] / 400.0),
            }
        )
    return samples


def summarize_move_quality(samples):
    losses = sorted(s["centipawn_loss"] for s in samples)
    return {
        "positions": len(samples),
        "avg_centipawn_loss": sum(losses) / len(losses),
        "median_centipawn_loss": losses[len(losses) // 2],
        "top1_match_rate": sum(1 for s in samples if s["rank"] == 1) / len(samples),
        "top3_match_rate": sum(
            1 for s in samples if s["rank"] is not None and s["rank"] <= 3
        )
        / len(samples),
        "value_mae": sum(abs(s["model_value"] - s["stockfish_value"]) for s in samples)
        / len(samples),
    }


def print_report(report):
    print("\nLadder results (weakest to strongest):")
    for level in report["levels"]:
        print(
            f"  vs {level['label']}: "
            f"{level['wins']}W {level['draws']}D {level['losses']}L "
            f"({level['score']:.1f}/{level['games']}), "
            f"white {level['white_score']:.1f}, black {level['black_score']:.1f}, "
            f"avg {level['avg_plies']:.0f} plies"
        )

    if report["estimated_rating"] is not None:
        print(
            f"\nEstimated calibrated rating: {report['estimated_rating']:.0f} "
            f"+/- {report['rating_stderr']:.0f}"
        )
    else:
        print("\nNo calibrated Elo levels were reachable; see ladder results above.")

    mq = report["move_quality"]
    print("\nMove quality vs Stockfish (depth-limited analysis):")
    print(f"  positions analysed: {mq['positions']}")
    print(f"  avg centipawn loss: {mq['avg_centipawn_loss']:.1f}")
    print(f"  median centipawn loss: {mq['median_centipawn_loss']:.1f}")
    print(f"  top-1 move match rate: {mq['top1_match_rate']:.1%}")
    print(f"  top-3 move match rate: {mq['top3_match_rate']:.1%}")
    print(f"  value head MAE vs Stockfish eval: {mq['value_mae']:.3f}")


def main():
    config = Config()
    device = get_device()
    model = load_checkpoint(CHECKPOINT_PATH, device, config)

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {device}")

    levels = run_adaptive_ladder(config.stockfish_path, model, device)
    calibrated = [level for level in levels if "elo" in level["level"]]
    rating = fit_rating(calibrated) if calibrated else None
    rating_se = rating_standard_error(rating, calibrated) if calibrated else None

    positions = load_or_create_positions(
        POSITIONS_FILE, MOVE_QUALITY_POSITIONS, POSITION_SEED, MIN_PLY, MAX_PLY
    )

    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    move_quality = summarize_move_quality(
        evaluate_move_quality(engine, model, device, positions, MOVE_QUALITY_DEPTH)
    )
    engine.quit()

    report = {
        "checkpoint": CHECKPOINT_PATH,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "estimated_rating": rating,
        "rating_stderr": rating_se,
        "levels": levels,
        "move_quality": move_quality,
    }

    print_report(report)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
