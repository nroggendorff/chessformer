import concurrent.futures
import math
import multiprocessing as mp
import random

import pecan as pc
from data_generation import pin_to_next_cpu
from oracle import Oracle
from tree_search import mcts_policy_step

ELO_EVAL_ANCHOR_SPREAD = (-1, 0, 1)

DEPTH_RATING_TABLE = [
    (1, 400),
    (2, 700),
    (3, 1000),
    (4, 1300),
    (5, 1600),
    (6, 1900),
    (7, 2200),
    (8, 2500),
]

_EVAL_ORACLE = None


def rating_for_depth(depth):
    for d, rating in DEPTH_RATING_TABLE:
        if d == depth:
            return rating
    return DEPTH_RATING_TABLE[-1][1]


def nearest_depth_index(rating):
    return min(
        range(len(DEPTH_RATING_TABLE)),
        key=lambda i: abs(DEPTH_RATING_TABLE[i][1] - rating),
    )


def random_opening_moves(game_index, plies):
    rng = random.Random(game_index)
    board = pc.Board()
    moves = []
    for _ in range(plies):
        if board.outcome() is not None:
            break
        move = rng.choice(board.legal_moves)
        board.push(move)
        moves.append(move)
    return moves


def expected_score(rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def binomial_z_score(wins, draws, games, baseline=0.5):
    if games == 0:
        return 0.0
    smoothed_n = games + 2
    smoothed_score = (wins + 0.5 * draws + 1) / smoothed_n
    se = math.sqrt(smoothed_score * (1 - smoothed_score) / smoothed_n)
    return ((wins + 0.5 * draws) / games - baseline) / se


def fit_rating(calibrated_results, lo=-3000.0, hi=4000.0, iters=80):
    if not calibrated_results:
        return None

    total_actual = sum(r["score"] for r in calibrated_results) + 0.5

    for _ in range(iters):
        mid = (lo + hi) / 2
        total_expected = sum(
            r["games"] * expected_score(mid, r["level"]["elo"])
            for r in calibrated_results
        ) + (1.0 * expected_score(mid, mid))

        if total_expected < total_actual:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def rating_standard_error(rating, calibrated_results, max_se=600.0):
    if rating is None or not calibrated_results:
        return None

    information = sum(
        r["games"]
        * expected_score(rating, r["level"]["elo"])
        * (1 - expected_score(rating, r["level"]["elo"]))
        for r in calibrated_results
    )
    if information <= 0:
        return max_se
    return min(max_se, 400 / (math.log(10) * math.sqrt(information)))


def play_eval_game(
    oracle,
    model,
    device,
    config,
    model_is_white,
    max_moves,
    oracle_depth,
    mcts_simulations=None,
    opening_moves=(),
    adjudication_depth=6,
):
    board = pc.Board()
    for move in opening_moves:
        board.push(move)
    mover = pc.WHITE if model_is_white else pc.BLACK
    plies = len(opening_moves)
    for _ in range(max(0, max_moves - plies)):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == mover:
            moves, _ = mcts_policy_step(
                [board],
                model,
                device,
                num_simulations=mcts_simulations or config.inference_mcts_simulations,
                sims_per_wave=config.mcts_sims_per_wave,
                c_puct=config.mcts_c_puct,
                temperature=0.0,
                target_batch_size=config.mcts_target_batch_size,
                max_batch_size=config.mcts_max_batch_size,
            )
            board.push(moves[0])
        else:
            board.push(oracle.play(board, depth=oracle_depth))
        plies += 1

    outcome = board.outcome(claim_draw=True)
    timed_out = outcome is None
    if timed_out:
        raw = oracle.score_cp(board, depth=adjudication_depth)
        cp = raw if board.turn == mover else -raw
        score = 1.0 if cp > 150 else 0.0 if cp < -150 else 0.5
    else:
        score = 0.5 if outcome.winner is None else float(outcome.winner == mover)
    return {"score": score, "plies": plies, "timed_out": timed_out}


def eval_worker_init(cpu_counter, cpu_lock):
    global _EVAL_ORACLE
    pin_to_next_cpu(cpu_counter, cpu_lock)
    _EVAL_ORACLE = Oracle()


def eval_worker_play_move(fen, depth):
    board = pc.Board(fen)
    return _EVAL_ORACLE.play(board, depth=depth).uci()


def eval_worker_timeout_score(fen, mover_is_white, depth=6):
    board = pc.Board(fen)
    raw = _EVAL_ORACLE.score_cp(board, depth=depth)
    cp = raw if board.turn == (pc.WHITE if mover_is_white else pc.BLACK) else -raw
    return 1.0 if cp > 150 else 0.0 if cp < -150 else 0.5


def play_all_anchor_games(
    model,
    device,
    config,
    anchor_depths,
    games_per_anchor,
    max_moves,
    max_workers,
    mcts_simulations=None,
    random_opening_plies=0,
    adjudication_depth=6,
):
    ctx = mp.get_context("spawn")
    total_games = games_per_anchor * len(anchor_depths)
    boards = [pc.Board() for _ in range(total_games)]
    model_is_white = [i % 2 == 0 for i in range(total_games)]
    anchor_of_game = [i // games_per_anchor for i in range(total_games)]
    finished = [False] * total_games
    plies = [0] * total_games

    if random_opening_plies > 0:
        for i in range(total_games):
            for move in random_opening_moves(i, random_opening_plies):
                boards[i].push(move)
                plies[i] += 1
                if boards[i].is_game_over(claim_draw=True):
                    finished[i] = True

    model.eval()
    pools = [
        concurrent.futures.ProcessPoolExecutor(
            max_workers=min(
                max(1, max_workers // len(anchor_depths)), games_per_anchor
            ),
            mp_context=ctx,
            initializer=eval_worker_init,
            initargs=(ctx.Value("i", 0), ctx.Lock()),
        )
        for _ in anchor_depths
    ]
    try:
        for _ in range(max_moves):
            active = [i for i in range(total_games) if not finished[i]]
            if not active:
                break

            learner_idx = [
                i
                for i in active
                if boards[i].turn == (pc.WHITE if model_is_white[i] else pc.BLACK)
            ]
            engine_idx = [i for i in active if i not in learner_idx]

            if learner_idx:
                moves, _ = mcts_policy_step(
                    [boards[i] for i in learner_idx],
                    model,
                    device,
                    num_simulations=mcts_simulations
                    or config.inference_mcts_simulations,
                    sims_per_wave=config.mcts_sims_per_wave,
                    c_puct=config.mcts_c_puct,
                    temperature=0.0,
                    target_batch_size=config.mcts_target_batch_size,
                    max_batch_size=config.mcts_max_batch_size,
                )
                for i, move in zip(learner_idx, moves):
                    boards[i].push(move)
                    plies[i] += 1
                    if boards[i].is_game_over(claim_draw=True):
                        finished[i] = True

            if engine_idx:
                futures = {
                    i: pools[anchor_of_game[i]].submit(
                        eval_worker_play_move,
                        boards[i].fen(),
                        anchor_depths[anchor_of_game[i]],
                    )
                    for i in engine_idx
                }
                for i, future in futures.items():
                    boards[i].push_uci(future.result())
                    plies[i] += 1
                    if boards[i].is_game_over(claim_draw=True):
                        finished[i] = True

        outcomes = [board.outcome(claim_draw=True) for board in boards]
        timeout_futures = {
            i: pools[anchor_of_game[i]].submit(
                eval_worker_timeout_score,
                boards[i].fen(),
                model_is_white[i],
                adjudication_depth,
            )
            for i, outcome in enumerate(outcomes)
            if outcome is None
        }

        results = [
            {
                "score": 0.0,
                "games": games_per_anchor,
                "level": {"elo": rating_for_depth(d), "depth": d},
            }
            for d in anchor_depths
        ]
        for i, outcome in enumerate(outcomes):
            mover = pc.WHITE if model_is_white[i] else pc.BLACK
            score = (
                timeout_futures[i].result()
                if outcome is None
                else (0.5 if outcome.winner is None else float(outcome.winner == mover))
            )
            results[anchor_of_game[i]]["score"] += score
    finally:
        for pool in pools:
            pool.shutdown(wait=False)

    return results


def adaptive_eval_anchors(config, state):
    center = state.get(
        "last_elo",
        state.get("elo_ema", rating_for_depth(config.self_play_oracle_depth)),
    )
    idx = nearest_depth_index(center)
    depths = sorted(
        {
            DEPTH_RATING_TABLE[max(0, min(len(DEPTH_RATING_TABLE) - 1, idx + spread))][
                0
            ]
            for spread in ELO_EVAL_ANCHOR_SPREAD
        }
    )
    return depths


def estimate_elo(model, device, config, state):
    model.eval()
    anchor_depths = adaptive_eval_anchors(config, state)
    games_per_anchor = max(2, config.elo_eval_games // len(anchor_depths))

    results = play_all_anchor_games(
        model,
        device,
        config,
        anchor_depths,
        games_per_anchor,
        config.elo_eval_max_moves,
        config.max_workers,
        mcts_simulations=config.elo_eval_mcts_simulations,
        random_opening_plies=config.elo_eval_random_plies,
        adjudication_depth=config.elo_eval_adjudication_depth,
    )

    elo = fit_rating(results)
    se = rating_standard_error(elo, results)

    state["elo_ema"] = (
        elo
        if "elo_ema" not in state
        else config.elo_eval_ema_alpha * elo
        + (1 - config.elo_eval_ema_alpha) * state["elo_ema"]
    )
    state["last_elo"], state["last_se"] = elo, se
    return elo, state["elo_ema"]
