import asyncio
import atexit
import concurrent.futures
import multiprocessing as mp
import os
import gc
import math
import random
import signal
import threading

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from encoding import board_to_input, canonical_square, legal_moves_by_square_pair

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
STARTING_NON_KING_MATERIAL = 78

_ENGINE = None


def _daemon_run_in_background(coroutine, *, name=None, debug=None):
    future = concurrent.futures.Future()

    def background():
        try:
            asyncio.run(coroutine(future), debug=debug)
            future.cancel()
        except Exception as exc:
            future.set_exception(exc)

    threading.Thread(target=background, name=name, daemon=True).start()
    return future.result()


chess.engine.run_in_background = _daemon_run_in_background


def pin_to_next_cpu(cpu_counter, cpu_lock):
    if not hasattr(os, "sched_setaffinity"):
        return
    with cpu_lock:
        cpu_id = cpu_counter.value
        cpu_counter.value += 1
    try:
        os.sched_setaffinity(0, {cpu_id % os.cpu_count()})
    except OSError:
        pass


def worker_shutdown(timeout=5):
    if _ENGINE is None:
        return

    def _quit():
        try:
            _ENGINE.quit()
        except Exception:
            pass

    quit_thread = threading.Thread(target=_quit, daemon=True)
    quit_thread.start()
    quit_thread.join(timeout=timeout)
    if quit_thread.is_alive():
        try:
            os.kill(_ENGINE.transport.get_pid(), signal.SIGKILL)
        except Exception:
            pass


def worker_init(engine_path, hash_mb, cpu_counter, cpu_lock):
    global _ENGINE
    pin_to_next_cpu(cpu_counter, cpu_lock)
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _ENGINE.configure({"Hash": hash_mb, "Threads": 1})
    atexit.register(worker_shutdown)


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


def win_probability(score, ply):
    return score.wdl(model="sf", ply=ply).expectation()


def move_scores(infos, board, temperature, cp_scale=100.0):
    cp_scores = {
        info["pv"][0]: info["score"].pov(board.turn).score(mate_score=3000)
        for info in infos
        if "pv" in info and len(info["pv"]) > 0
    }
    best = max(cp_scores.values(), default=0)
    return {
        move: math.exp((cp - best) / (cp_scale * temperature))
        for move, cp in cp_scores.items()
    }


def endgame_weight(board, scale):
    return 1 + scale * (
        1
        - min(
            sum(
                len(board.pieces(piece_type, color)) * value
                for piece_type, value in PIECE_VALUES.items()
                for color in chess.COLORS
            ),
            STARTING_NON_KING_MATERIAL,
        )
        / STARTING_NON_KING_MATERIAL
    )


def position_label(win_prob, scores, board, weight=1.0):
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
        "value": 2 * win_prob - 1,
        "policy_weight": weight,
        "value_weight": weight,
    }


def generate_game(
    engine,
    max_moves=60,
    depth_range=(2, 8),
    drive_depth=3,
    sample_moves=None,
    drive_multipv=8,
    sample_multipv=12,
    endgame_weight_scale=2.0,
    policy_temperature=0.5,
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
        depth = (
            round(random.triangular(depth_range[0], depth_range[1], depth_range[1]))
            if is_sample
            else drive_depth
        )
        infos = analyse_full_policy(
            engine, board, depth, multipv=sample_multipv if is_sample else drive_multipv
        )
        if not infos:
            break

        scores = move_scores(infos, board, policy_temperature)
        if is_sample:
            samples.append(
                position_label(
                    win_probability(infos[0]["score"].pov(board.turn), board.ply()),
                    scores,
                    board,
                    weight=(depth / depth_range[1]) ** 2
                    * endgame_weight(board, endgame_weight_scale),
                )
            )
        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
        )

    return samples


def worker_generate_games(
    num_games,
    max_moves=60,
    depth_range=(2, 8),
    drive_depth=3,
    sample_moves=None,
    drive_multipv=8,
    sample_multipv=12,
    endgame_weight_scale=2.0,
    policy_temperature=0.5,
):
    return [
        sample
        for _ in range(num_games)
        for sample in generate_game(
            _ENGINE,
            max_moves,
            depth_range,
            drive_depth,
            sample_moves,
            drive_multipv,
            sample_multipv,
            endgame_weight_scale,
            policy_temperature,
        )
    ]


def generate_pretrain_data(config):
    total_games, games_per_task, max_workers = (
        config.pretrain_games,
        config.pretrain_games_per_task,
        config.max_workers,
    )
    task_game_counts = [games_per_task] * (total_games // games_per_task)
    if total_games % games_per_task:
        task_game_counts.append(total_games % games_per_task)

    ctx = mp.get_context("spawn")
    cpu_counter, cpu_lock = ctx.Value("i", 0), ctx.Lock()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(
            config.stockfish_path,
            config.pretrain_hash_mb,
            cpu_counter,
            cpu_lock,
        ),
    ) as executor:
        futures = [
            executor.submit(
                worker_generate_games,
                count,
                config.pretrain_max_moves,
                (config.pretrain_traj_depth, config.pretrain_depth),
                config.pretrain_drive_depth,
                config.pretrain_sample_moves,
                config.pretrain_drive_multipv,
                config.pretrain_sample_multipv,
                config.pretrain_endgame_weight,
                config.pretrain_policy_temperature,
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
