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
    if not list(board.legal_moves):
        return None
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=4)
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


def position_label(infos, board):
    scores = move_scores(infos, board)
    legal_pairs = np.array(
        list(legal_moves_by_square_pair(board).keys()), dtype=np.uint8
    )
    if scores:
        total = sum(scores.values())
        policy_pairs = np.array(
            [
                (
                    canonical_square(move.from_square, board),
                    canonical_square(move.to_square, board),
                )
                for move in scores
            ],
            dtype=np.uint8,
        )
        policy_probs = np.array(
            [sc / total for sc in scores.values()], dtype=np.float32
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
        1.0,
    )


def analyse_position(engine, board, depth):
    infos = analyse_multipv(engine, board, depth)
    return position_label(infos, board) if infos else None


def play_low_depth_game(engine, max_moves=60, traj_depth=2):
    board, positions = chess.Board(), []
    for _ in range(max_moves):
        if board.is_game_over():
            break
        infos = analyse_multipv(engine, board, traj_depth)
        if not infos:
            break

        positions.append(board.copy())
        scores = move_scores(infos, board)
        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
            if scores
            else random.choice(list(board.legal_moves))
        )

    positions.append(board.copy())
    return positions


def generate_game(engine, max_moves=60, traj_depth=2, label_depth=8, sample_moves=None):
    positions = play_low_depth_game(engine, max_moves, traj_depth)
    sampled = random.sample(
        positions, min(sample_moves or len(positions), len(positions))
    )
    return [
        sample
        for board in sampled
        for sample in [analyse_position(engine, board, label_depth)]
        if sample is not None
    ]


def worker_generate_games(
    engine_path, num_games, max_moves=60, traj_depth=2, label_depth=8, sample_moves=None
):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    samples = [
        sample
        for _ in range(num_games)
        for sample in generate_game(
            engine, max_moves, traj_depth, label_depth, sample_moves
        )
    ]
    engine.quit()
    return samples


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
        for i in tqdm(
            range(0, len(task_game_counts), max_workers), desc="Parallel Gen"
        ):
            futures = [
                executor.submit(
                    worker_generate_games,
                    config.stockfish_path,
                    count,
                    config.pretrain_max_moves,
                    config.pretrain_traj_depth,
                    config.pretrain_depth,
                    config.pretrain_sample_moves,
                )
                for count in task_game_counts[i : i + max_workers]
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    replay.extend_pretrain(f.result())
                except Exception:
                    pass
            gc.collect()
