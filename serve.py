import argparse

from flask import Flask, jsonify, request

import pecan as pc
from config import Config, default_checkpoint_path, get_device
from model import load_checkpoint
from tree_search import mcts_move

app = Flask(__name__)
state = {}


def board_to_json(board):
    cells = []
    for sq in range(pc.NUM_SQUARES):
        piece = board.board[sq]
        cells.append(
            None
            if piece is None
            else ("w" if piece[0] == pc.WHITE else "b") + pc.PIECE_SYMBOLS[piece[1]]
        )
    outcome = board.outcome(claim_draw=True)
    return {
        "board": cells,
        "turn": "w" if board.turn == pc.WHITE else "b",
        "in_check": board.is_check(),
        "game_over": outcome is not None,
        "termination": outcome.termination if outcome is not None else None,
        "winner": (
            None
            if outcome is None or outcome.winner is None
            else ("w" if outcome.winner == pc.WHITE else "b")
        ),
        "history": [move.uci() for move in board.move_stack],
        "legal_moves": [move.uci() for move in board.legal_moves],
    }


@app.route("/api/state")
def api_state():
    return jsonify(board_to_json(state["board"]))


@app.route("/api/new", methods=["POST"])
def api_new():
    state["board"] = pc.Board()
    return jsonify(board_to_json(state["board"]))


@app.route("/api/undo", methods=["POST"])
def api_undo():
    board = state["board"]
    for _ in range(2):
        if board.move_stack:
            board.pop()
    return jsonify(board_to_json(board))


@app.route("/api/move", methods=["POST"])
def api_move():
    board = state["board"]
    if board.is_game_over(claim_draw=True):
        return jsonify(board_to_json(board))

    move_text = request.json.get("move", "").strip()
    try:
        board.push_uci(move_text)
    except (ValueError, IndexError, KeyError):
        return (
            jsonify({"error": "illegal or unparseable move", **board_to_json(board)}),
            400,
        )

    if not board.is_game_over(claim_draw=True):
        board.push(mcts_move(board, state["model"], state["device"], state["config"]))

    return jsonify(board_to_json(board))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=default_checkpoint_path())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    config = Config()
    device = get_device()
    state["board"] = pc.Board()
    state["model"] = load_checkpoint(args.checkpoint, device, config)
    state["config"] = config
    state["device"] = device

    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
