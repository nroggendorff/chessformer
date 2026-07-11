import argparse
import json
import os

import chess
import chess.engine
import chess.svg
from flask import Flask, jsonify, render_template, request

from config import Config, default_checkpoint_path, get_device
from model import load_checkpoint
from policy import batched_policy_step

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))
state = {}

PIECE_NAMES = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}

PIECE_SVGS = {
    symbol: chess.svg.piece(chess.Piece.from_symbol(symbol), size=45)
    for symbol in "PNBRQKpnbrqk"
}


def bot_move(board, model, device):
    moves, _, _, _ = batched_policy_step([board], model, device, temperature=0.0)
    return moves[0]


def analyse(board):
    info = state["engine"].analyse(board, chess.engine.Limit(depth=state["eval_depth"]))
    if "score" in info:
        return info["score"]
    if board.is_checkmate():
        return chess.engine.PovScore(chess.engine.Mate(-1), board.turn)
    return chess.engine.PovScore(chess.engine.Cp(0), board.turn)


def eval_display(score):
    pov = score.pov(chess.WHITE)
    return (
        {"mate": pov.mate()} if pov.is_mate() else {"cp": pov.score(mate_score=10000)}
    )


def classify_move(cp_before, cp_after):
    loss = max(0, cp_before - cp_after)
    if loss >= 300:
        return "blunder", loss
    if loss >= 120:
        return "mistake", loss
    if loss >= 50:
        return "inaccuracy", loss
    return None, loss


def game_status(board):
    if board.is_checkmate():
        return ("Black" if board.turn else "White") + " wins by Checkmate"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Draw (Insufficient Material)"
    if board.can_claim_draw():
        return "Draw Available"
    return ("White" if board.turn == chess.WHITE else "Black") + " to Move"


def capture_square(board, move):
    if board.is_en_passant(move):
        return move.to_square + (-8 if board.turn == chess.WHITE else 8)
    return move.to_square


def piece_to_str(piece):
    if not piece:
        return "No"
    return f"{'White' if piece.color == chess.WHITE else 'Black'} {PIECE_NAMES[piece.piece_type]}"


def describe_move(board, move):
    info: dict[str, object] = {
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
    }

    if board.is_capture(move):
        info["capture_square"] = chess.square_name(capture_square(board, move))

    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(
            move.from_square
        )
        info["castle"] = True
        info["rook_from"] = chess.square_name(chess.square(7 if kingside else 0, rank))
        info["rook_to"] = chess.square_name(chess.square(5 if kingside else 3, rank))

    if move.promotion:
        info["promotion"] = chess.piece_symbol(move.promotion)

    return info


def captured_pieces(board):
    starting_counts = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
    }
    return {
        key: [
            (
                chess.piece_symbol(piece_type).upper()
                if color == chess.WHITE
                else chess.piece_symbol(piece_type)
            )
            for piece_type, count in starting_counts.items()
            for _ in range(count - len(board.pieces(piece_type, color)))
        ]
        for color, key in ((chess.WHITE, "white"), (chess.BLACK, "black"))
    }


def move_history(board, qualities):
    replay = chess.Board()
    history = []
    for i, move in enumerate(board.move_stack):
        san = replay.san(move)
        quality = qualities[i] if i < len(qualities) else {}
        history.append(
            {
                "san": san,
                "quality": quality.get("quality"),
                "cp_loss": quality.get("cp_loss"),
            }
        )
        replay.push(move)
    return history


def legal_move_map(board):
    targets: dict[str, list[str]] = {}
    for move in board.legal_moves:
        targets.setdefault(chess.square_name(move.from_square), []).append(
            chess.square_name(move.to_square)
        )
    return {square: sorted(set(squares)) for square, squares in targets.items()}


def serialize(board, score):
    pieces = [
        {"square": chess.square_name(sq), "symbol": piece.symbol()}
        for sq, piece in board.piece_map().items()
    ]

    check_square = None
    if board.is_check():
        king_sq = board.king(board.turn)
        if king_sq is not None:
            check_square = chess.square_name(king_sq)

    return {
        "svg": chess.svg.board(
            board,
            lastmove=board.peek() if board.move_stack else None,
            size=480,
            coordinates=False,
            colors={"square light": "#f0d9b5", "square dark": "#b58863"},
        ),
        "pieces": pieces,
        "fen": board.fen(),
        "status": game_status(board),
        "game_over": board.is_game_over(claim_draw=True),
        "eval": eval_display(score),
        "check_square": check_square,
        "legal_moves": legal_move_map(board),
        "captured": captured_pieces(board),
        "history": move_history(board, state["move_qualities"]),
    }


def get_move_log_info(board, move):
    return {
        "ply": board.ply() + 1,
        "piece": piece_to_str(board.piece_at(move.from_square)),
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "capture": piece_to_str(board.piece_at(capture_square(board, move))),
    }


def print_move_log(log_info, score_after):
    pov = score_after.pov(chess.WHITE)
    eval_str = (
        f"Mate in {pov.mate()}"
        if pov.is_mate()
        else str(round(pov.score(mate_score=10000) / 100.0, 1))
    )
    print(json.dumps({**log_info, "eval": eval_str}, indent=4))


def push_and_log(board, move):
    mover = board.turn
    cp_before = state["last_score"].pov(mover).score(mate_score=10000)
    log_info = get_move_log_info(board, move)
    move_info = describe_move(board, move)

    board.push(move)
    score = analyse(board)
    print_move_log(log_info, score)

    quality, cp_loss = classify_move(
        cp_before, score.pov(mover).score(mate_score=10000)
    )
    state["move_qualities"].append({"quality": quality, "cp_loss": cp_loss})
    state["last_score"] = score
    return {**move_info, "quality": quality, "cp_loss": cp_loss}, score


@app.route("/")
def index():
    return render_template("index.html", piece_svg_json=json.dumps(PIECE_SVGS))


@app.route("/api/state")
def api_state():
    return jsonify(serialize(state["board"], state["last_score"]))


@app.route("/api/new", methods=["POST"])
def api_new():
    state["board"] = chess.Board()
    state["move_qualities"] = []
    state["last_score"] = analyse(state["board"])
    return jsonify(serialize(state["board"], state["last_score"]))


@app.route("/api/undo", methods=["POST"])
def api_undo():
    board = state["board"]
    popped = 0
    while popped < 2 and board.move_stack:
        board.pop()
        popped += 1
    state["move_qualities"] = state["move_qualities"][
        : len(state["move_qualities"]) - popped
    ]
    state["last_score"] = analyse(board)
    return jsonify(serialize(board, state["last_score"]))


@app.route("/api/move", methods=["POST"])
def api_move():
    board = state["board"]
    if board.is_game_over(claim_draw=True):
        return jsonify({**serialize(board, state["last_score"]), "moves": []})

    move_text = request.json.get("move", "").strip()
    try:
        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            raise ValueError
    except ValueError:
        try:
            move = board.parse_san(move_text)
        except ValueError:
            return (
                jsonify(
                    {
                        "error": "Illegal or unparseable move",
                        **serialize(board, state["last_score"]),
                    }
                ),
                400,
            )

    human_move, score = push_and_log(board, move)
    moves = [human_move]

    if not board.is_game_over(claim_draw=True):
        bot_reply, score = push_and_log(
            board, bot_move(board, state["model"], state["device"])
        )
        moves.append(bot_reply)

    return jsonify({**serialize(board, score), "moves": moves})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=default_checkpoint_path())
    parser.add_argument("--stockfish-path", default="/usr/games/stockfish")
    parser.add_argument("--eval-depth", type=int, default=14)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    config = Config(stockfish_path=args.stockfish_path)
    device = get_device()
    state["board"] = chess.Board()
    state["model"] = load_checkpoint(args.checkpoint, device, config)
    state["device"] = device
    state["eval_depth"] = args.eval_depth
    state["engine"] = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    state["move_qualities"] = []
    state["last_score"] = analyse(state["board"])

    try:
        app.run(host=args.host, port=args.port, threaded=False)
    finally:
        state["engine"].quit()


if __name__ == "__main__":
    main()
