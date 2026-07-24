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
import traceback

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


_GAME_COUNTER = None


def worker_init(engine_path, hash_mb, cpu_counter, cpu_lock, game_counter):
    global _ENGINE, _GAME_COUNTER
    gc.set_threshold(100000, 50, 50)
    pin_to_next_cpu(cpu_counter, cpu_lock)
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _ENGINE.configure({"Hash": hash_mb, "Threads": 1})
    _GAME_COUNTER = game_counter
    atexit.register(worker_shutdown)


def analyse_full_policy(
    engine, board, depth, multipv=None, nodes=None, legal_moves=None
):
    legal_moves = list(board.legal_moves) if legal_moves is None else legal_moves
    if not legal_moves:
        return None
    infos = engine.analyse(
        board,
        chess.engine.Limit(depth=depth, nodes=nodes),
        multipv=min(multipv or len(legal_moves), len(legal_moves)),
    )
    return infos if isinstance(infos, list) else [infos]


def analyse_converged(
    engine,
    board,
    max_depth,
    multipv,
    nodes=None,
    min_depth=8,
    stability=3,
    score_margin=25,
    legal_moves=None,
):
    legal_moves = list(board.legal_moves) if legal_moves is None else legal_moves
    if not legal_moves:
        return None, max_depth
    multipv = min(multipv, len(legal_moves))
    by_depth, last_best, last_score, streak = {}, None, None, 0
    with engine.analysis(
        board, chess.engine.Limit(depth=max_depth, nodes=nodes), multipv=multipv
    ) as analysis:
        for info in analysis:
            if "pv" not in info or "depth" not in info or "score" not in info:
                continue
            depth = info["depth"]
            by_depth.setdefault(depth, {})[info.get("multipv", 1)] = info
            if len(by_depth[depth]) < multipv:
                continue
            top = by_depth[depth][1]
            top_move = top["pv"][0]
            top_score = top["score"].pov(board.turn).score(mate_score=100000)
            stable = top_move == last_best and (
                last_score is not None and abs(top_score - last_score) <= score_margin
            )
            streak = streak + 1 if stable else 1
            last_best, last_score = top_move, top_score
            if depth >= min_depth and streak >= stability:
                break
    if not by_depth:
        return None, max_depth
    final_depth = max(by_depth)
    return [
        by_depth[final_depth][i] for i in sorted(by_depth[final_depth])
    ], final_depth


def win_probability(score, ply):
    return score.wdl(model="sf", ply=ply).expectation()


def score_to_value(pov_score, scale=400.0):
    return math.tanh(pov_score.score(mate_score=100000) / scale)


def move_win_probs(infos, board):
    return {
        info["pv"][0]: win_probability(info["score"].pov(board.turn), board.ply())
        for info in infos
        if "pv" in info and len(info["pv"]) > 0
    }


def move_scores(win_probs, temperature):
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


def position_label(value, scores, board, weight=1.0, legal_moves=None):
    if legal_moves is None:
        legal_moves = list(board.legal_moves)
    pair_scores = {}
    for move, score in scores.items():
        key = (
            canonical_square(move.from_square, board),
            canonical_square(move.to_square, board),
        )
        pair_scores[key] = pair_scores.get(key, 0.0) + score

    total = sum(pair_scores.values()) or 1.0
    return {
        "board_input": np.array(
            board_to_input(board, legal_moves=legal_moves), dtype=np.uint8
        ),
        "legal_pairs": np.array(
            list(legal_moves_by_square_pair(board, legal_moves=legal_moves).keys()),
            dtype=np.uint8,
        ).reshape(-1, 2),
        "policy_pairs": np.array(list(pair_scores.keys()), dtype=np.uint8).reshape(
            -1, 2
        ),
        "policy_probs": np.array(
            [sc / total for sc in pair_scores.values()], dtype=np.float32
        ),
        "value": value,
        "policy_weight": weight,
        "value_weight": weight,
    }


def child_value_rows(board, infos, weight):
    rows = []
    for info in infos:
        if not info.get("pv"):
            continue
        move = info["pv"][0]
        child = board.copy()
        child.push(move)
        child_legal_moves = list(child.legal_moves)
        rows.append(
            {
                "board_input": np.array(
                    board_to_input(child, legal_moves=child_legal_moves), dtype=np.uint8
                ),
                "legal_pairs": np.array(
                    list(
                        legal_moves_by_square_pair(
                            child, legal_moves=child_legal_moves
                        ).keys()
                    ),
                    dtype=np.uint8,
                ).reshape(-1, 2),
                "policy_pairs": np.zeros((0, 2), dtype=np.uint8),
                "policy_probs": np.zeros((0,), dtype=np.float32),
                "value": -score_to_value(info["score"].pov(board.turn)),
                "policy_weight": 0.0,
                "value_weight": weight,
            }
        )
    return rows


def _should_sample_position(ply, win_probs, sample_ply_ramp, max_win_prob, min_entropy):
    if not win_probs or len(win_probs) < 2:
        return False
    best_wp = max(win_probs.values())
    if best_wp > max_win_prob:
        return False
    total = sum(win_probs.values())
    if total <= 0:
        return False
    probs = [wp / total for wp in win_probs.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 1e-10)
    if entropy < min_entropy:
        return False
    keep_prob = 1.0 if sample_ply_ramp <= 0 else min(1.0, (ply + 1) / sample_ply_ramp)
    return random.random() < keep_prob


def generate_game(
    engine,
    max_moves=60,
    depth_range=(8, 16),
    drive_depth=3,
    sample_moves=None,
    drive_multipv=8,
    sample_multipv=8,
    endgame_weight_scale=2.0,
    policy_temperature=0.06,
    drive_temperature=0.3,
    node_cap=None,
    sample_ply_ramp=10,
    max_sample_win_prob=0.85,
    min_sample_entropy=0.3,
    sample_stability=3,
    sample_score_margin=25,
):
    board = chess.Board()
    sample_plies = set(
        random.sample(
            range(max_moves),
            min(sample_moves if sample_moves is not None else max_moves, max_moves),
        )
    )
    samples = []
    for ply in range(max_moves):
        if board.is_game_over():
            break
        is_sample = ply in sample_plies
        legal_moves = list(board.legal_moves)
        if is_sample:
            infos, depth = analyse_converged(
                engine,
                board,
                depth_range[1],
                sample_multipv,
                nodes=node_cap,
                min_depth=depth_range[0],
                stability=sample_stability,
                score_margin=sample_score_margin,
                legal_moves=legal_moves,
            )
        else:
            infos, depth = (
                analyse_full_policy(
                    engine,
                    board,
                    drive_depth,
                    multipv=drive_multipv,
                    nodes=node_cap,
                    legal_moves=legal_moves,
                ),
                drive_depth,
            )
        if not infos:
            break

        win_probs = move_win_probs(infos, board)
        scores = move_scores(
            win_probs, policy_temperature if is_sample else drive_temperature
        )
        if is_sample and _should_sample_position(
            board.ply(),
            win_probs,
            sample_ply_ramp,
            max_sample_win_prob,
            min_sample_entropy,
        ):
            weight = (depth / depth_range[1]) ** 2 * endgame_weight(
                board, endgame_weight_scale
            )
            samples.append(
                position_label(
                    score_to_value(infos[0]["score"].pov(board.turn)),
                    scores,
                    board,
                    weight=weight,
                    legal_moves=legal_moves,
                )
            )
            samples.extend(child_value_rows(board, infos, weight))
        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
        )

    return samples


def worker_generate_games(
    num_games,
    max_moves=60,
    depth_range=(8, 16),
    drive_depth=3,
    sample_moves=None,
    drive_multipv=8,
    sample_multipv=8,
    endgame_weight_scale=2.0,
    policy_temperature=0.06,
    drive_temperature=0.3,
    node_cap=None,
    sample_ply_ramp=10,
    max_sample_win_prob=0.85,
    min_sample_entropy=0.3,
    sample_stability=3,
    sample_score_margin=25,
):
    samples = []
    for _ in range(num_games):
        try:
            samples.extend(
                generate_game(
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
                    sample_ply_ramp,
                    max_sample_win_prob,
                    min_sample_entropy,
                    sample_stability,
                    sample_score_margin,
                )
            )
        except Exception:
            tqdm.write(
                f"skipping a game that raised an error:\n{traceback.format_exc()}"
            )
        if _GAME_COUNTER is not None:
            with _GAME_COUNTER.get_lock():
                _GAME_COUNTER.value += 1
    return samples


def generate_pretrain_data(config):
    total_games, max_workers = config.pretrain_games, config.max_workers

    chunk_size = min(config.pretrain_chunk_games, total_games) if total_games else 0
    task_game_counts = (
        [chunk_size] * (total_games // chunk_size)
        + ([total_games % chunk_size] if total_games % chunk_size else [])
        if chunk_size
        else []
    )

    ctx = mp.get_context("spawn")
    cpu_counter, cpu_lock = ctx.Value("i", 0), ctx.Lock()
    game_counter = ctx.Value("i", 0)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(
            config.stockfish_path,
            config.pretrain_hash_mb,
            cpu_counter,
            cpu_lock,
            game_counter,
        ),
    ) as executor:
        pending = {
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
                config.pretrain_sample_ply_ramp,
                config.pretrain_max_sample_win_prob,
                config.pretrain_min_sample_entropy,
                config.pretrain_sample_stability,
                config.pretrain_sample_score_margin,
            )
            for count in task_game_counts
        }
        with tqdm(
            total=total_games, desc="Pretrain data generation", unit="games"
        ) as pbar:
            completed, failed, yielded = 0, 0, 0
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                pbar.update(game_counter.value - pbar.n)
                for f in done:
                    try:
                        samples = f.result()
                    except Exception:
                        failed += 1
                        tqdm.write(
                            f"worker chunk {completed} failed "
                            f"({failed}/{len(task_game_counts)} chunks so far):\n"
                            f"{traceback.format_exc()}"
                        )
                        samples = []
                    yielded += len(samples)
                    yield from samples
                    completed += 1
                    if completed % max_workers == 0:
                        gc.collect()
            pbar.update(game_counter.value - pbar.n)

    if yielded == 0 and task_game_counts:
        raise RuntimeError(
            f"generate_pretrain_data produced no samples: all {failed}/{len(task_game_counts)} "
            "worker chunks failed. Check stockfish_path and the errors logged above."
        )
