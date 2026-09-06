import atexit
import concurrent.futures
import contextlib
import math
import multiprocessing as mp
import random

import chess
import chess.engine

from data_generation import pin_to_next_cpu
from tree_search import game_over, game_result, mcts_policy_step

WEAK_ANCHOR_LADDER = (
    {"name": "random", "elo": -515, "engine_prob": 0.0},
    {"name": "mix10", "elo": -240, "engine_prob": 0.10},
    {"name": "mix25", "elo": -35, "engine_prob": 0.25},
    {"name": "mix50", "elo": 350, "engine_prob": 0.50},
    {"name": "mix75", "elo": 685, "engine_prob": 0.75},
    {"name": "weak", "elo": 1160, "engine_prob": 1.0},
)
WEAK_ANCHOR_SKILL = 0
WEAK_ANCHOR_NODES = 1
UCI_ELO_MIN = 1320
UCI_ELO_MAX = 3190

EVAL_OPENING_LINES = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f8c5"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5"),
    ("g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"),
)

_EVAL_ENGINE = None
_EVAL_ANCHOR = None


def clamp_uci_elo(engine, elo):
    option = engine.options.get("UCI_Elo")
    return elo if option is None else max(option.min, min(option.max, elo))


def anchor_ladder(config):
    rungs = [dict(rung) for rung in WEAK_ANCHOR_LADDER]
    elo = UCI_ELO_MIN
    while elo <= UCI_ELO_MAX:
        rungs.append(
            {"name": f"sf{elo}", "elo": elo, "engine_prob": 1.0, "uci_elo": elo}
        )
        elo += config.elo_eval_anchor_step
    return rungs


def anchor_for_elo(config, target):
    return min(anchor_ladder(config), key=lambda rung: abs(rung["elo"] - target))


def configure_anchor(engine, anchor):
    if "uci_elo" in anchor:
        engine.configure(
            {
                "UCI_LimitStrength": True,
                "UCI_Elo": clamp_uci_elo(engine, anchor["uci_elo"]),
            }
        )
        return None
    engine.configure({"UCI_LimitStrength": False, "Skill Level": WEAK_ANCHOR_SKILL})
    return chess.engine.Limit(nodes=WEAK_ANCHOR_NODES)


def anchor_move(engine, anchor, limit, board, movetime):
    if engine is None or random.random() >= anchor["engine_prob"]:
        legal = list(board.legal_moves)
        return random.choice(legal) if legal else None
    return engine.play(board, limit or chess.engine.Limit(time=movetime)).move


def opening_moves_for_game(game_index, plies=None):
    line = EVAL_OPENING_LINES[game_index % len(EVAL_OPENING_LINES)]
    return line if plies is None else line[: max(0, plies)]


def expected_score(rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def binomial_z_score(wins, draws, games, baseline=0.5):
    if games <= 1:
        return 0.0
    losses = max(0, games - wins - draws)
    smoothed_n = games + 2
    score = (wins + 0.5 * draws + 1) / smoothed_n
    variance = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * score**2
        + 2 * (0.5 - score) ** 2
    ) / smoothed_n
    se = math.sqrt(max(variance, 0.25 / smoothed_n) / smoothed_n)
    return (score - baseline) / se


def fit_rating(calibrated_results, lo=-4000.0, hi=4000.0, iters=80):
    if not calibrated_results:
        return None

    prior_level = min(r["level"]["elo"] for r in calibrated_results)
    total_actual = sum(r["score"] for r in calibrated_results) + 0.5

    for _ in range(iters):
        mid = (lo + hi) / 2
        total_expected = sum(
            r["games"] * expected_score(mid, r["level"]["elo"])
            for r in calibrated_results
        ) + expected_score(mid, prior_level)

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
    engine,
    model,
    device,
    config,
    model_is_white,
    max_moves,
    limit,
    mcts_simulations=None,
    opening_moves=(),
    adjudication_depth=10,
):
    board = chess.Board()
    opening_moves = opening_moves[: max(0, max_moves)]
    for move in opening_moves:
        board.push_uci(move)
    mover = chess.WHITE if model_is_white else chess.BLACK
    plies = len(opening_moves)
    for _ in range(max(0, max_moves - plies)):
        if game_over(board):
            break
        if board.turn == mover:
            moves, _ = mcts_policy_step(
                [board],
                model,
                device,
                num_simulations=mcts_simulations or config.inference_mcts_simulations,
                sims_per_wave=config.mcts_sims_per_wave,
                c_puct=config.mcts_c_puct,
                fpu_reduction=config.mcts_fpu_reduction,
                temperature=0.0,
                target_batch_size=config.mcts_target_batch_size,
                max_batch_size=config.mcts_max_batch_size,
            )
            board.push(moves[0])
        else:
            board.push(engine.play(board, limit).move)
        plies += 1

    finished, winner = game_result(board)
    timed_out = not finished
    if timed_out:
        cp = (
            engine.analyse(board, chess.engine.Limit(depth=adjudication_depth))["score"]
            .pov(mover)
            .score(mate_score=10000)
        )
        score = 1.0 if cp > 150 else 0.0 if cp < -150 else 0.5
    else:
        score = 0.5 if winner is None else float(winner == mover)
    return {"score": score, "plies": plies, "timed_out": timed_out}


def _eval_worker_shutdown():
    if _EVAL_ENGINE is not None:
        try:
            _EVAL_ENGINE.quit()
        except Exception:
            pass


def eval_worker_init(engine_path, anchor, cpu_counter, cpu_lock):
    global _EVAL_ENGINE, _EVAL_ANCHOR
    pin_to_next_cpu(cpu_counter, cpu_lock)
    _EVAL_ANCHOR = dict(anchor)
    _EVAL_ANCHOR["limit"] = None
    if anchor["engine_prob"] <= 0.0:
        return
    _EVAL_ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _EVAL_ANCHOR["limit"] = configure_anchor(_EVAL_ENGINE, anchor)
    atexit.register(_eval_worker_shutdown)


def eval_worker_play_move(fen, movetime):
    board = chess.Board(fen)
    move = anchor_move(
        _EVAL_ENGINE, _EVAL_ANCHOR, _EVAL_ANCHOR["limit"], board, movetime
    )
    return None if move is None else move.uci()


_ADJUDICATION_ENGINE = None


def _adjudication_shutdown():
    if _ADJUDICATION_ENGINE is not None:
        try:
            _ADJUDICATION_ENGINE.quit()
        except Exception:
            pass


def adjudication_worker_init(engine_path):
    global _ADJUDICATION_ENGINE
    _ADJUDICATION_ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    atexit.register(_adjudication_shutdown)


def adjudicate_position(fen, mover_is_white, depth, margin):
    score = (
        _ADJUDICATION_ENGINE.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))[
            "score"
        ]
        .pov(chess.WHITE if mover_is_white else chess.BLACK)
        .score(mate_score=10000)
    )
    return 1.0 if score > margin else 0.0 if score < -margin else 0.5


def adjudicate_unresolved(engine_path, positions, depth, margin, max_workers=1):
    if not positions:
        return []
    workers = max(1, min(max_workers, len(positions)))
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=adjudication_worker_init,
        initargs=(engine_path,),
    ) as pool:
        futures = [
            pool.submit(adjudicate_position, fen, white, depth, margin)
            for fen, white in positions
        ]
        return [f.result() for f in futures]


def play_all_anchor_games(
    engine_path,
    model,
    device,
    config,
    anchors,
    games_per_anchor,
    max_moves,
    movetime,
    max_workers,
    mcts_simulations=None,
    random_opening_plies=0,
    adjudication_depth=10,
    adjudication_margin=150,
):
    with chess.engine.SimpleEngine.popen_uci(engine_path) as probe_engine:
        anchors = [
            (
                {**anchor, "elo": clamp_uci_elo(probe_engine, anchor["uci_elo"])}
                if "uci_elo" in anchor
                else dict(anchor)
            )
            for anchor in anchors
        ]

    ctx = mp.get_context("spawn")
    total_games = games_per_anchor * len(anchors)
    boards = [chess.Board() for _ in range(total_games)]
    model_is_white = [i % 2 == 0 for i in range(total_games)]
    anchor_of_game = [i // games_per_anchor for i in range(total_games)]
    finished = [False] * total_games
    plies = [0] * total_games

    model.eval()
    with contextlib.ExitStack() as stack:
        pools = [
            stack.enter_context(
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(
                        max(1, max_workers // len(anchors)), games_per_anchor
                    ),
                    mp_context=ctx,
                    initializer=eval_worker_init,
                    initargs=(engine_path, anchor, ctx.Value("i", 0), ctx.Lock()),
                )
            )
            for anchor in anchors
        ]

        for _ in range(max_moves):
            active = [i for i in range(total_games) if not finished[i]]
            if not active:
                break

            learner_idx = [
                i
                for i in active
                if boards[i].turn == (chess.WHITE if model_is_white[i] else chess.BLACK)
            ]
            learner_set = set(learner_idx)
            engine_idx = [i for i in active if i not in learner_set]

            if learner_idx:
                warm = [i for i in learner_idx if plies[i] < random_opening_plies]
                warm_set = set(warm)
                for subset, temp in (
                    (warm, 1.0),
                    ([i for i in learner_idx if i not in warm_set], 0.0),
                ):
                    if not subset:
                        continue
                    moves, _ = mcts_policy_step(
                        [boards[i] for i in subset],
                        model,
                        device,
                        num_simulations=mcts_simulations
                        or config.inference_mcts_simulations,
                        sims_per_wave=config.mcts_sims_per_wave,
                        c_puct=config.mcts_c_puct,
                        fpu_reduction=config.mcts_fpu_reduction,
                        temperature=temp,
                        target_batch_size=config.mcts_target_batch_size,
                        max_batch_size=config.mcts_max_batch_size,
                    )
                    for i, move in zip(subset, moves):
                        boards[i].push(move)
                        plies[i] += 1
                        if game_over(boards[i]):
                            finished[i] = True

            if engine_idx:
                futures = {
                    i: pools[anchor_of_game[i]].submit(
                        eval_worker_play_move, boards[i].fen(), movetime
                    )
                    for i in engine_idx
                }
                for i, future in futures.items():
                    uci = future.result()
                    if uci is None:
                        finished[i] = True
                        continue
                    boards[i].push_uci(uci)
                    plies[i] += 1
                    if game_over(boards[i]):
                        finished[i] = True

    outcomes = [game_result(board) for board in boards]
    pending = [i for i, (done, _) in enumerate(outcomes) if not done]
    adjudicated = dict(
        zip(
            pending,
            adjudicate_unresolved(
                engine_path,
                [(boards[i].fen(), model_is_white[i]) for i in pending],
                adjudication_depth,
                adjudication_margin,
                max_workers=max_workers,
            ),
        )
    )

    results = [
        {"score": 0.0, "games": games_per_anchor, "level": anchor} for anchor in anchors
    ]
    for i, (done, winner) in enumerate(outcomes):
        mover = chess.WHITE if model_is_white[i] else chess.BLACK
        results[anchor_of_game[i]]["score"] += (
            (0.5 if winner is None else float(winner == mover))
            if done
            else adjudicated[i]
        )

    return results


def adaptive_eval_anchors(config, state):
    ladder = anchor_ladder(config)
    center = state.get("last_elo", state.get("elo_ema", config.elo_eval_anchor))
    pinned = state.get("anchor_center")
    if pinned is not None and abs(center - pinned) < config.elo_eval_recenter_margin:
        center = pinned
    state["anchor_center"] = center

    nearest = min(range(len(ladder)), key=lambda i: abs(ladder[i]["elo"] - center))
    width = min(config.elo_eval_anchor_rungs, len(ladder))
    start = max(0, min(nearest - width // 2, len(ladder) - width))
    return ladder[start : start + width]


def estimate_elo(model, device, config, state):
    model.eval()
    anchors = adaptive_eval_anchors(config, state)
    games_per_anchor = max(2, config.elo_eval_games // len(anchors))

    results = play_all_anchor_games(
        config.stockfish_path,
        model,
        device,
        config,
        anchors,
        games_per_anchor,
        config.elo_eval_max_moves,
        config.elo_eval_movetime,
        config.max_workers,
        mcts_simulations=config.elo_eval_mcts_simulations,
        random_opening_plies=config.elo_eval_random_plies,
        adjudication_depth=config.elo_eval_adjudication_depth,
        adjudication_margin=config.elo_eval_adjudication_margin,
    )

    elo = fit_rating(results)
    se = rating_standard_error(elo, results)
    scored = sum(r["score"] for r in results)
    played = sum(r["games"] for r in results)

    state["last_censored"] = scored <= 0.0 or scored >= played
    state["elo_ema"] = (
        elo
        if "elo_ema" not in state
        else config.elo_eval_ema_alpha * elo
        + (1 - config.elo_eval_ema_alpha) * state["elo_ema"]
    )
    state["last_elo"], state["last_se"] = elo, se
    return elo, state["elo_ema"]
