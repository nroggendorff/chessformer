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


def analyse_full_policy(engine, board, depth, multipv=None, nodes=None):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    infos = engine.analyse(
        board,
        chess.engine.Limit(depth=depth, nodes=nodes),
        multipv=min(multipv or len(legal_moves), len(legal_moves)),
    )
    return infos if isinstance(infos, list) else [infos]


def win_probability(score, ply):
    return score.wdl(model="sf", ply=ply).expectation()


def move_scores(infos, board, temperature):
    win_probs = {
        info["pv"][0]: win_probability(info["score"].pov(board.turn), board.ply())
        for info in infos
        if "pv" in info and len(info["pv"]) > 0
    }
    best = max(win_probs.values(), default=0.0)
    return {move: math.exp((wp - best) / temperature) for move, wp in win_probs.items()}


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


def _should_sample_position(ply, win_probs, min_sample_ply, max_win_prob, min_entropy):
    if ply < min_sample_ply:
        return False
    if not win_probs:
        return False
    best_wp = max(win_probs.values())
    if best_wp > max_win_prob:
        return False
    if len(win_probs) < 2:
        return False
    total = sum(win_probs.values())
    if total <= 0:
        return False
    probs = [wp / total for wp in win_probs.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 1e-10)
    return entropy >= min_entropy


def generate_game(
    engine,
    max_moves=60,
    depth_range=(4, 16),
    drive_depth=3,
    sample_moves=None,
    drive_multipv=8,
    sample_multipv=8,
    endgame_weight_scale=2.0,
    policy_temperature=0.06,
    drive_temperature=0.3,
    node_cap=None,
    min_sample_ply=10,
    max_sample_win_prob=0.85,
    min_sample_entropy=0.3,
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
            engine,
            board,
            depth,
            multipv=sample_multipv if is_sample else drive_multipv,
            nodes=node_cap,
        )
        if not infos:
            break

        scores = move_scores(
            infos, board, policy_temperature if is_sample else drive_temperature
        )
        win_probs = {
            info["pv"][0]: win_probability(info["score"].pov(board.turn), board.ply())
            for info in infos
            if "pv" in info and len(info["pv"]) > 0
        }
        if is_sample and _should_sample_position(
            board.ply(),
            win_probs,
            min_sample_ply,
            max_sample_win_prob,
            min_sample_entropy,
        ):
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
    depth_range=(4, 16),
    drive_depth=3,
    sample_moves=None,
    drive_multipv=8,
    sample_multipv=8,
    endgame_weight_scale=2.0,
    policy_temperature=0.06,
    drive_temperature=0.3,
    node_cap=None,
    min_sample_ply=10,
    max_sample_win_prob=0.85,
    min_sample_entropy=0.3,
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
            drive_temperature,
            node_cap,
            min_sample_ply,
            max_sample_win_prob,
            min_sample_entropy,
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
        futures = {
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
                config.pretrain_drive_temperature,
                config.pretrain_node_cap,
                config.pretrain_min_sample_ply,
                config.pretrain_max_sample_win_prob,
                config.pretrain_min_sample_entropy,
            ): count
            for count in task_game_counts
        }
        with tqdm(
            total=total_games, desc="Pretrain data generation", unit="games"
        ) as pbar:
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    samples = f.result()
                except Exception:
                    samples = []
                yield from samples
                pbar.update(futures[f])
                if i % max_workers == 0:
                    gc.collect()
