import concurrent.futures
import gc
import math
import random

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from encoding import board_to_tokens, canonical_square, legal_moves_by_square_pair


def analyse_multipv(engine, board, depth):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    infos = engine.analyse(
        board, chess.engine.Limit(depth=depth), multipv=len(legal_moves)
    )
    if not isinstance(infos, list):
        infos = [infos]
    return infos or None


def move_scores(infos, board):
    return {
        info["pv"][0]: math.exp(
            max(min(info["score"].pov(board.turn).score(mate_score=10000), 1000), -1000)
            / 250.0
        )
        for info in infos
        if "pv" in info and len(info["pv"]) > 0
    }


def position_label(infos, board, scores, weight=1.0):
    legal_pairs = np.array(
        list(legal_moves_by_square_pair(board).keys()), dtype=np.uint8
    )
    if scores:
        pair_scores = {}
        for move, score in scores.items():
            key = (
                canonical_square(move.from_square, board),
                canonical_square(move.to_square, board),
            )
            pair_scores[key] = pair_scores.get(key, 0.0) + score

        total = sum(pair_scores.values())
        policy_pairs = np.array(list(pair_scores.keys()), dtype=np.uint8)
        policy_probs = np.array(
            [sc / total for sc in pair_scores.values()], dtype=np.float32
        )
    else:
        policy_pairs = legal_pairs
        policy_probs = np.full(
            len(legal_pairs), 1.0 / len(legal_pairs), dtype=np.float32
        )

    value = math.tanh(infos[0]["score"].pov(board.turn).score(mate_score=10000) / 400.0)
    return (
        np.array(board_to_tokens(board), dtype=np.uint8),
        legal_pairs,
        policy_pairs,
        policy_probs,
        value,
        weight,
    )


def generate_game(engine, max_moves=60, depth_range=(2, 8), sample_moves=None):
    board, samples = chess.Board(), []
    for _ in range(max_moves):
        if board.is_game_over():
            break
        depth = random.randint(*depth_range)
        infos = analyse_multipv(engine, board, depth)
        if not infos:
            break

        scores = move_scores(infos, board)
        samples.append(
            position_label(infos, board, scores, weight=depth / depth_range[1])
        )
        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
            if scores
            else random.choice(list(board.legal_moves))
        )

    return random.sample(samples, min(sample_moves or len(samples), len(samples)))


def worker_generate_games(
    engine_path, num_games, max_moves=60, depth_range=(2, 8), sample_moves=None
):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        return [
            sample
            for _ in range(num_games)
            for sample in generate_game(engine, max_moves, depth_range, sample_moves)
        ]
    finally:
        engine.quit()


def generate_pretrain_data(config, replay):
    total_games, games_per_task, max_workers = (
        config.pretrain_games,
        config.pretrain_games_per_task,
        config.max_workers,
    )
    task_game_counts = [games_per_task] * (total_games // games_per_task)
    if total_games % games_per_task:
        task_game_counts.append(total_games % games_per_task)

    print(f"Generating {total_games} games using {max_workers} CPU cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                worker_generate_games,
                config.stockfish_path,
                count,
                config.pretrain_max_moves,
                (config.pretrain_traj_depth, config.pretrain_depth),
                config.pretrain_sample_moves,
            )
            for count in task_game_counts
        ]
        for i, f in enumerate(
            tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Parallel Gen",
            )
        ):
            try:
                replay.extend_pretrain(f.result())
            except Exception:
                pass
            if i % max_workers == 0:
                gc.collect()
