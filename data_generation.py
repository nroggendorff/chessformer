import concurrent.futures
import gc
import math
import multiprocessing as mp
import os
import random
import traceback

import numpy as np
from tqdm import tqdm

import pecan as pc
from encoding import board_to_input, canon_square, legal_moves_by_square_pair
from oracle import Oracle

STARTING_NON_KING_MATERIAL = pc.NON_KING_STARTING_MATERIAL

_ORACLE = None


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


def worker_init(cpu_counter, cpu_lock):
    global _ORACLE
    gc.set_threshold(100000, 50, 50)
    pin_to_next_cpu(cpu_counter, cpu_lock)
    _ORACLE = Oracle()


def score_to_value(score_cp, scale=400.0):
    return math.tanh(score_cp / scale)


def win_probability(score_cp, scale=400.0):
    return (score_to_value(score_cp, scale) + 1.0) / 2.0


def move_scores(win_probs, temperature):
    best = max(win_probs.values(), default=0.0)
    return {move: math.exp((wp - best) / temperature) for move, wp in win_probs.items()}


def endgame_weight(board, scale):
    material = sum(
        pc.VALUES[p[1]] for p in board.board if p is not None and p[1] != pc.KING
    )
    return 1 + scale * (
        1 - min(material, STARTING_NON_KING_MATERIAL) / STARTING_NON_KING_MATERIAL
    )


def _top_k(scored_moves, k):
    return sorted(scored_moves, key=lambda pair: pair[1], reverse=True)[: max(1, k)]


def position_label(
    value, scores, board, weight=1.0, legal_moves=None, include_policy=True
):
    if legal_moves is None:
        legal_moves = board.legal_moves
    mover = board.turn
    pair_scores = {}
    for move, score in scores.items():
        key = (
            canon_square(move.from_square, mover),
            canon_square(move.to_square, mover),
        )
        pair_scores[key] = pair_scores.get(key, 0.0) + score

    total = sum(pair_scores.values()) or 1.0
    return {
        "board_input": np.array(board_to_input(board), dtype=np.int64),
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
        "policy_weight": weight if include_policy else 0.0,
        "value_weight": weight,
    }


def _should_include_policy(ply, win_probs, sample_ply_ramp, max_win_prob, max_entropy):
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
    if entropy > max_entropy:
        return False
    keep_prob = 1.0 if sample_ply_ramp <= 0 else min(1.0, (ply + 1) / sample_ply_ramp)
    return random.random() < keep_prob


def generate_game(
    oracle,
    max_moves=60,
    sample_depth=6,
    drive_depth=2,
    sample_moves=None,
    drive_top_k=8,
    sample_top_k=6,
    endgame_weight_scale=2.0,
    policy_temperature=0.06,
    drive_temperature=0.3,
    node_cap=None,
    sample_ply_ramp=8,
    max_sample_win_prob=0.97,
    max_sample_entropy=1.5,
):
    board = pc.Board()
    sample_plies = set(
        random.sample(range(max_moves), min(sample_moves or max_moves, max_moves))
    )
    samples = []
    for ply in range(max_moves):
        if board.outcome() is not None:
            break
        is_sample = ply in sample_plies
        legal_moves = board.legal_moves
        if not legal_moves:
            break

        result = oracle.search(
            board, depth=sample_depth if is_sample else drive_depth, node_cap=node_cap
        )
        root_scores = result["root_scores"]
        if not root_scores:
            break

        kept = _top_k(root_scores, sample_top_k if is_sample else drive_top_k)
        win_probs = {move: win_probability(score) for move, score in kept}
        scores = move_scores(
            win_probs, policy_temperature if is_sample else drive_temperature
        )

        if is_sample:
            weight = (result["depth"] / sample_depth) ** 2 * endgame_weight(
                board, endgame_weight_scale
            )
            samples.append(
                position_label(
                    score_to_value(result["score"]),
                    scores,
                    board,
                    weight=weight,
                    legal_moves=legal_moves,
                    include_policy=_should_include_policy(
                        board.ply(),
                        win_probs,
                        sample_ply_ramp,
                        max_sample_win_prob,
                        max_sample_entropy,
                    ),
                )
            )
        board.push(
            random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
        )

    return samples


def worker_generate_games(
    num_games,
    max_moves=60,
    sample_depth=6,
    drive_depth=2,
    sample_moves=None,
    drive_top_k=8,
    sample_top_k=6,
    endgame_weight_scale=2.0,
    policy_temperature=0.06,
    drive_temperature=0.3,
    node_cap=None,
    sample_ply_ramp=8,
    max_sample_win_prob=0.97,
    max_sample_entropy=1.5,
):
    samples = []
    for _ in range(num_games):
        try:
            samples.extend(
                generate_game(
                    _ORACLE,
                    max_moves,
                    sample_depth,
                    drive_depth,
                    sample_moves,
                    drive_top_k,
                    sample_top_k,
                    endgame_weight_scale,
                    policy_temperature,
                    drive_temperature,
                    node_cap,
                    sample_ply_ramp,
                    max_sample_win_prob,
                    max_sample_entropy,
                )
            )
        except Exception:
            tqdm.write(
                f"skipping a game that raised an error:\n{traceback.format_exc()}"
            )
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

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(cpu_counter, cpu_lock),
    ) as executor:
        pending = {
            executor.submit(
                worker_generate_games,
                count,
                config.pretrain_max_moves,
                config.pretrain_depth,
                config.pretrain_drive_depth,
                config.pretrain_sample_moves,
                config.pretrain_drive_top_k,
                config.pretrain_sample_top_k,
                config.pretrain_endgame_weight,
                config.pretrain_policy_temperature,
                config.pretrain_drive_temperature,
                config.pretrain_node_cap,
                config.pretrain_sample_ply_ramp,
                config.pretrain_max_sample_win_prob,
                config.pretrain_max_sample_entropy,
            )
            for count in task_game_counts
        }
        with tqdm(
            total=len(task_game_counts), desc="Pretrain data generation", unit="chunks"
        ) as pbar:
            completed, failed, yielded = 0, 0, 0
            for f in concurrent.futures.as_completed(pending):
                try:
                    samples = f.result()
                except Exception:
                    failed += 1
                    tqdm.write(
                        f"worker chunk {completed} failed "
                        f"({failed}/{len(task_game_counts)} chunks so far):\n{traceback.format_exc()}"
                    )
                    samples = []
                yielded += len(samples)
                yield from samples
                completed += 1
                pbar.update(1)
                if completed % max_workers == 0:
                    gc.collect()

    if yielded == 0 and task_game_counts:
        raise RuntimeError(
            f"generate_pretrain_data produced no samples: all {failed}/{len(task_game_counts)} "
            "worker chunks failed. See the errors logged above."
        )
