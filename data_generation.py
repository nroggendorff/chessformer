import concurrent.futures
import gc
import math
import multiprocessing
import random

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from encoding import board_to_tokens


def generate_game(engine, max_moves=60, depth=3):
    board, samples = chess.Board(), []

    for _ in range(max_moves):
        if board.is_game_over():
            break
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=4)
        if not isinstance(infos, list):
            infos = [infos]
        if not infos or not list(board.legal_moves):
            break

        scores = {}
        for info in infos:
            if "pv" in info and len(info["pv"]) > 0:
                move = info["pv"][0]
                sc = info["score"].pov(board.turn).score(mate_score=10000)
                scores[move] = math.exp(max(min(sc, 1000), -1000) / 100.0)

        policy_grid = np.zeros((64, 64), dtype=np.float32)
        if scores:
            total = sum(scores.values())
            for move, sc in scores.items():
                policy_grid[move.from_square, move.to_square] = sc / total
        else:
            legal_moves = list(board.legal_moves)
            uniform = 1.0 / len(legal_moves)
            for move in legal_moves:
                policy_grid[move.from_square, move.to_square] = uniform

        value = math.tanh(infos[0]["score"].white().score(mate_score=10000) / 400.0)
        samples.append(
            (np.array(board_to_tokens(board), dtype=np.uint8), policy_grid, value)
        )

        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
            if scores
            else random.choice(list(board.legal_moves))
        )

    return samples


def worker_generate_games(engine_path, num_games, max_moves=60, depth=3):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    samples = [
        sample
        for _ in range(num_games)
        for sample in generate_game(engine, max_moves, depth)
    ]
    engine.quit()
    return samples


def generate_pretrain_data(
    stockfish_path,
    total_games,
    games_per_task=50,
    max_workers=None,
    max_moves=60,
    depth=3,
):
    max_workers = max_workers or multiprocessing.cpu_count()
    task_game_counts = [games_per_task] * (total_games // games_per_task)
    if total_games % games_per_task:
        task_game_counts.append(total_games % games_per_task)

    print(f"Generating {total_games} expert games using {max_workers} CPU cores...")

    pretrain_samples = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(
            range(0, len(task_game_counts), max_workers), desc="Parallel Gen"
        ):
            futures = [
                executor.submit(
                    worker_generate_games, stockfish_path, count, max_moves, depth
                )
                for count in task_game_counts[i : i + max_workers]
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    pretrain_samples.extend(f.result())
                except Exception:
                    pass
            gc.collect()

    return pretrain_samples
