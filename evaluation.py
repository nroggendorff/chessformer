import math

import chess
import chess.engine

from policy import batched_policy_step

ELO_EVAL_ANCHOR_SPREAD = (-200, 0, 200)


def clamp_uci_elo(engine, elo):
    option = engine.options.get("UCI_Elo")
    return elo if option is None else max(option.min, min(option.max, elo))


def expected_score(rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


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


def rating_standard_error(rating, calibrated_results):
    if not rating or not calibrated_results:
        return None

    information = sum(
        r["games"]
        * expected_score(rating, r["level"]["elo"])
        * (1 - expected_score(rating, r["level"]["elo"]))
        for r in calibrated_results
    )
    return (
        400 / (math.log(10) * math.sqrt(information))
        if information > 0
        else float("inf")
    )


def play_eval_game(engine, model, device, model_is_white, max_moves, limit):
    board = chess.Board()
    mover = chess.WHITE if model_is_white else chess.BLACK
    plies = 0
    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == mover:
            moves, _, _, _ = batched_policy_step(
                [board], model, device, temperature=0.0
            )
            board.push(moves[0])
        else:
            board.push(engine.play(board, limit).move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    score = (
        0.5
        if outcome is None or outcome.winner is None
        else float(outcome.winner == mover)
    )
    return {"score": score, "plies": plies, "timed_out": outcome is None}


def play_anchor_games(
    engine_path, model, device, anchor, num_games, max_moves, movetime
):
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        elo = clamp_uci_elo(engine, anchor)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        score = sum(
            play_eval_game(
                engine,
                model,
                device,
                i % 2 == 0,
                max_moves,
                chess.engine.Limit(time=movetime),
            )["score"]
            for i in range(num_games)
        )
    return {"score": score, "games": num_games, "level": {"elo": elo}}


def estimate_elo(model, device, config, state):
    model.eval()
    anchors = [config.elo_eval_anchor + spread for spread in ELO_EVAL_ANCHOR_SPREAD]
    games_per_anchor = max(2, config.elo_eval_games // len(anchors))

    results = [
        play_anchor_games(
            config.stockfish_path,
            model,
            device,
            anchor,
            games_per_anchor,
            config.elo_eval_max_moves,
            config.elo_eval_movetime,
        )
        for anchor in anchors
    ]

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
