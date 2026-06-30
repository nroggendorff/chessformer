import concurrent.futures
import gc
import math
import random

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from encoding import board_to_tokens, canonical_square, legal_moves_by_square_pair


def analyse_multipv(engine, board, depth, multipv_cap=10):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    infos = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=min(len(legal_moves), multipv_cap),
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


def move_q_targets(infos, board):
    pair_targets = {}
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        move = info["pv"][0]
        q = math.tanh(info["score"].pov(board.turn).score(mate_score=10000) / 400.0)
        key = (
            canonical_square(move.from_square, board),
            canonical_square(move.to_square, board),
        )
        pair_targets[key] = max(pair_targets.get(key, -1.0), q)
    return pair_targets


def position_label(infos, board, weight=1.0):
    legal_pairs = np.array(
        list(legal_moves_by_square_pair(board).keys()), dtype=np.uint8
    )
    q_targets = move_q_targets(infos, board)
    q_pairs = (
        np.array(list(q_targets.keys()), dtype=np.uint8)
        if q_targets
        else np.zeros((0, 2), dtype=np.uint8)
    )
    q_values = np.array(list(q_targets.values()), dtype=np.float32)

    return (
        np.array(board_to_tokens(board), dtype=np.uint8),
        legal_pairs,
        q_pairs,
        q_values,
        weight,
    )


def generate_game(
    engine, max_moves=60, depth_range=(2, 8), sample_moves=None, multipv_cap=10
):
    board, samples = chess.Board(), []
    for _ in range(max_moves):
        if board.is_game_over():
            break
        depth = round(random.triangular(*depth_range, depth_range[0]))
        infos = analyse_multipv(engine, board, depth, multipv_cap)
        if not infos:
            break

        scores = move_scores(infos, board)
        samples.append(
            position_label(infos, board, weight=(depth / depth_range[1]) ** 2)
        )
        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
            if scores
            else random.choice(list(board.legal_moves))
        )

    return random.sample(samples, min(sample_moves or len(samples), len(samples)))


def worker_generate_games(
    engine_path,
    num_games,
    max_moves=60,
    depth_range=(2, 8),
    sample_moves=None,
    hash_mb=128,
    multipv_cap=10,
):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Hash": hash_mb})
    try:
        return [
            sample
            for _ in range(num_games)
            for sample in generate_game(
                engine, max_moves, depth_range, sample_moves, multipv_cap
            )
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
                config.pretrain_hash_mb,
                config.pretrain_multipv,
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
