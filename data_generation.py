import concurrent.futures
import gc
import math
import random

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from encoding import board_to_input, canonical_square, legal_moves_by_square_pair


def analyse_value(engine, board, depth):
    return engine.analyse(board, chess.engine.Limit(depth=depth))


def analyse_full_policy(engine, board, depth, multipv=None):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    infos = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=min(multipv or len(legal_moves), len(legal_moves)),
    )
    return infos if isinstance(infos, list) else [infos]


def move_scores(infos, board):
    return {
        info["pv"][0]: math.exp(
            max(min(info["score"].pov(board.turn).score(mate_score=10000), 1000), -1000)
            / 250.0
        )
        for info in infos
        if "pv" in info and len(info["pv"]) > 0
    }


def position_label(value_score, scores, board, weight=1.0):
    pair_scores = {}
    for move, score in scores.items():
        key = (
            canonical_square(move.from_square, board),
            canonical_square(move.to_square, board),
        )
        pair_scores[key] = pair_scores.get(key, 0.0) + score

    total = sum(pair_scores.values()) or 1.0
    return {
        "board_input": np.array(board_to_input(board), dtype=np.uint8),
        "legal_pairs": np.array(
            list(legal_moves_by_square_pair(board).keys()), dtype=np.uint8
        ),
        "policy_pairs": np.array(list(pair_scores.keys()), dtype=np.uint8),
        "policy_probs": np.array(
            [sc / total for sc in pair_scores.values()], dtype=np.float32
        ),
        "value": math.tanh(value_score / 400.0),
        "policy_weight": weight,
        "value_weight": weight,
    }


def generate_game(
    engine,
    max_moves=60,
    depth_range=(2, 8),
    policy_depth=3,
    sample_moves=None,
    drive_multipv=8,
):
    board = chess.Board()
    sample_plies = set(
        random.sample(range(max_moves), min(sample_moves or max_moves, max_moves))
    )
    samples = []
    for ply in range(max_moves):
        if board.is_game_over():
            break
        is_sample = ply in sample_plies
        policy_infos = analyse_full_policy(
            engine, board, policy_depth, multipv=None if is_sample else drive_multipv
        )
        if not policy_infos:
            break

        scores = move_scores(policy_infos, board)
        if is_sample:
            depth = round(random.triangular(*depth_range, depth_range[1]))
            value_score = (
                analyse_value(engine, board, depth)["score"]
                .pov(board.turn)
                .score(mate_score=10000)
            )
            samples.append(
                position_label(
                    value_score, scores, board, weight=(depth / depth_range[1]) ** 2
                )
            )
        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
        )

    return samples


def worker_generate_games(
    engine_path,
    num_games,
    max_moves=60,
    depth_range=(2, 8),
    policy_depth=3,
    sample_moves=None,
    hash_mb=128,
    drive_multipv=8,
):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Hash": hash_mb})
    try:
        return [
            sample
            for _ in range(num_games)
            for sample in generate_game(
                engine,
                max_moves,
                depth_range,
                policy_depth,
                sample_moves,
                drive_multipv,
            )
        ]
    finally:
        engine.quit()


def generate_pretrain_data(config):
    total_games, games_per_task, max_workers = (
        config.pretrain_games,
        config.pretrain_games_per_task,
        config.max_workers,
    )
    task_game_counts = [games_per_task] * (total_games // games_per_task)
    if total_games % games_per_task:
        task_game_counts.append(total_games % games_per_task)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                worker_generate_games,
                config.stockfish_path,
                count,
                config.pretrain_max_moves,
                (config.pretrain_traj_depth, config.pretrain_depth),
                config.pretrain_policy_depth,
                config.pretrain_sample_moves,
                config.pretrain_hash_mb,
                config.pretrain_drive_multipv,
            )
            for count in task_game_counts
        ]
        for i, f in enumerate(
            tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Pretrain data generation",
            )
        ):
            try:
                yield from f.result()
            except Exception:
                pass
            if i % max_workers == 0:
                gc.collect()
