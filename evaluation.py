import math

import chess
import chess.engine

from policy import batched_policy_step


def clamp_uci_elo(engine, elo):
    option = engine.options.get("UCI_Elo")
    return elo if option is None else max(option.min, min(option.max, elo))


def play_eval_game(engine, model, device, model_is_white, max_moves, limit):
    board = chess.Board()
    mover = chess.WHITE if model_is_white else chess.BLACK
    plies = 0
    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == mover:
            moves, _, _ = batched_policy_step([board], model, device, temperature=0.0)
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
    return {"score": score, "plies": plies}


def estimate_elo(model, device, config, state):
    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    anchor = clamp_uci_elo(engine, config.elo_eval_anchor)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": anchor})
    model.eval()

    score = sum(
        play_eval_game(
            engine,
            model,
            device,
            i % 2 == 0,
            config.elo_eval_max_moves,
            chess.engine.Limit(depth=config.elo_eval_depth),
        )["score"]
        for i in range(config.elo_eval_games)
    )
    engine.quit()

    clipped = min(max(score, 0.5), config.elo_eval_games - 0.5)
    elo = anchor + 400 * math.log10(clipped / (config.elo_eval_games - clipped))

    state["elo_ema"] = (
        elo
        if "elo_ema" not in state
        else config.elo_eval_ema_alpha * elo
        + (1 - config.elo_eval_ema_alpha) * state["elo_ema"]
    )
    return elo, state["elo_ema"]
