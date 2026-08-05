import atexit
import concurrent.futures
import contextlib
import math
import multiprocessing as mp

import chess
import chess.engine

from data_generation import pin_to_next_cpu
from tree_search import mcts_policy_step

ELO_EVAL_ANCHOR_SPREAD = (-200, 0, 200)
EVAL_OPENING_LINES = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f8c5"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5"),
    ("g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"),
)

_EVAL_ENGINE = None


def clamp_uci_elo(engine, elo):
    option = engine.options.get("UCI_Elo")
    return elo if option is None else max(option.min, min(option.max, elo))


def opening_moves_for_game(game_index, plies=None):
    line = EVAL_OPENING_LINES[game_index % len(EVAL_OPENING_LINES)]
    return line if plies is None else line[: max(0, plies)]


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
            board.push(engine.play(board, limit).move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    timed_out = outcome is None
    if timed_out:
        cp = (
            engine.analyse(board, chess.engine.Limit(depth=adjudication_depth))["score"]
            .pov(mover)
            .score(mate_score=10000)
        )
        score = 1.0 if cp > 150 else 0.0 if cp < -150 else 0.5
    else:
        score = 0.5 if outcome.winner is None else float(outcome.winner == mover)
    return {"score": score, "plies": plies, "timed_out": timed_out}


def _eval_worker_shutdown():
    if _EVAL_ENGINE is not None:
        try:
            _EVAL_ENGINE.quit()
        except Exception:
            pass


def eval_worker_init(engine_path, elo, cpu_counter, cpu_lock):
    global _EVAL_ENGINE
    pin_to_next_cpu(cpu_counter, cpu_lock)
    _EVAL_ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _EVAL_ENGINE.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    atexit.register(_eval_worker_shutdown)


def eval_worker_play_move(fen, movetime):
    return _EVAL_ENGINE.play(
        chess.Board(fen), chess.engine.Limit(time=movetime)
    ).move.uci()


def eval_worker_timeout_score(fen, mover_is_white, depth=10):
    cp = (
        _EVAL_ENGINE.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))["score"]
        .pov(chess.WHITE if mover_is_white else chess.BLACK)
        .score(mate_score=10000)
    )
    return 1.0 if cp > 150 else 0.0 if cp < -150 else 0.5


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
):
    with chess.engine.SimpleEngine.popen_uci(engine_path) as probe_engine:
        elos = [clamp_uci_elo(probe_engine, anchor) for anchor in anchors]

    ctx = mp.get_context("spawn")
    total_games = games_per_anchor * len(elos)
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
                    max_workers=min(max(1, max_workers // len(elos)), games_per_anchor),
                    mp_context=ctx,
                    initializer=eval_worker_init,
                    initargs=(engine_path, elo, ctx.Value("i", 0), ctx.Lock()),
                )
            )
            for elo in elos
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
            engine_idx = [i for i in active if i not in learner_idx]

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
                        temperature=temp,
                        target_batch_size=config.mcts_target_batch_size,
                        max_batch_size=config.mcts_max_batch_size,
                    )
                    for i, move in zip(subset, moves):
                        boards[i].push(move)
                        plies[i] += 1
                        if boards[i].is_game_over(claim_draw=True):
                            finished[i] = True

            if engine_idx:
                futures = {
                    i: pools[anchor_of_game[i]].submit(
                        eval_worker_play_move, boards[i].fen(), movetime
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
            {"score": 0.0, "games": games_per_anchor, "level": {"elo": elo}}
            for elo in elos
        ]
        for i, outcome in enumerate(outcomes):
            mover = chess.WHITE if model_is_white[i] else chess.BLACK
            score = (
                timeout_futures[i].result()
                if outcome is None
                else (0.5 if outcome.winner is None else float(outcome.winner == mover))
            )
            results[anchor_of_game[i]]["score"] += score

    return results


def adaptive_eval_anchors(config, state):
    center = state.get("last_elo", state.get("elo_ema", config.elo_eval_anchor))
    center = int(round(center / 50.0) * 50)
    return [center + spread for spread in ELO_EVAL_ANCHOR_SPREAD]


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
