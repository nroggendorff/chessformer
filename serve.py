import argparse
import json

import chess
import chess.engine
import chess.svg
import torch
from flask import Flask, jsonify, render_template, request
from safetensors.torch import load_file

from config import Config
from mcts import MCTSNode, batched_mcts_sim, root_converged, root_policy_from_visits
from model import ChessNet

app = Flask(__name__)
state = {}


def load_model(checkpoint_path, device, config):
    model = ChessNet(
        d_model=config.d_model, nhead=config.nhead, enc_layers=config.enc_layers
    ).to(device)
    model.load_state_dict(load_file(checkpoint_path, device=device.type))
    model.eval()
    return model


def bot_move(board, model, device, sims, cpuct=1.5):
    root = MCTSNode()
    batched_mcts_sim([root], [board], model, device, cpuct=cpuct, add_noise=False)
    for _ in range(sims - 1):
        if root_converged(root):
            break
        batched_mcts_sim([root], [board], model, device, cpuct=cpuct, add_noise=False)
    moves, probs = root_policy_from_visits(root, temperature=0)
    return moves[int(torch.argmax(probs).item())]


def analyse(board):
    info = state["engine"].analyse(board, chess.engine.Limit(depth=state["eval_depth"]))
    return info["score"]


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


def describe_move(board, move):
    info = {
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
    }

    if board.is_en_passant(move):
        ep_square = move.to_square + (-8 if board.turn == chess.WHITE else 8)
        info["capture_square"] = chess.square_name(ep_square)
    elif board.is_capture(move):
        info["capture_square"] = info["to"]

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
    result = {"white": [], "black": []}
    for color, key in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        for piece_type, count in starting_counts.items():
            missing = count - len(board.pieces(piece_type, color))
            symbol = (
                chess.piece_symbol(piece_type).upper()
                if color == chess.WHITE
                else chess.piece_symbol(piece_type)
            )
            result[key].extend([symbol] * missing)
    return result


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
    targets = {}
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
            colors={
                "square light": "#f0d9b5",
                "square dark": "#b58863",
            },
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


PIECE_SVGS = {
    symbol: chess.svg.piece(chess.Piece.from_symbol(symbol), size=45)
    for symbol in "PNBRQKpnbrqk"
}


@app.route("/")
def index():
    return render_template("index.html", piece_svg_json=json.dumps(PIECE_SVGS))


@app.route("/api/state")
def api_state():
    board = state["board"]
    score = analyse(board)
    state["last_score"] = score
    return jsonify(serialize(board, score))


@app.route("/api/new", methods=["POST"])
def api_new():
    state["board"] = chess.Board()
    state["move_qualities"] = []
    score = analyse(state["board"])
    state["last_score"] = score
    return jsonify(serialize(state["board"], score))


@app.route("/api/undo", methods=["POST"])
def api_undo():
    board = state["board"]
    popped = 0
    for _ in range(2):
        if board.move_stack:
            board.pop()
            popped += 1
    state["move_qualities"] = state["move_qualities"][
        : len(state["move_qualities"]) - popped
    ]
    score = analyse(board)
    state["last_score"] = score
    return jsonify(serialize(board, score))


@app.route("/api/move", methods=["POST"])
def api_move():
    board = state["board"]
    if board.is_game_over(claim_draw=True):
        score = state.get("last_score") or analyse(board)
        return jsonify({**serialize(board, score), "moves": []})

    move_text = request.json.get("move", "").strip()
    try:
        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            raise ValueError
    except ValueError:
        try:
            move = board.parse_san(move_text)
        except ValueError:
            score = state.get("last_score") or analyse(board)
            return (
                jsonify(
                    {"error": "Illegal or unparseable move", **serialize(board, score)}
                ),
                400,
            )

    score_before = state.get("last_score") or analyse(board)
    mover = board.turn
    cp_before = score_before.pov(mover).score(mate_score=10000)

    moves = [describe_move(board, move)]
    board.push(move)

    score = analyse(board)
    cp_after = score.pov(mover).score(mate_score=10000)
    quality, cp_loss = classify_move(cp_before, cp_after)
    moves[0]["quality"] = quality
    moves[0]["cp_loss"] = cp_loss
    state["move_qualities"].append({"quality": quality, "cp_loss": cp_loss})

    if not board.is_game_over(claim_draw=True):
        bot_mover = board.turn
        cp_before_bot = score.pov(bot_mover).score(mate_score=10000)

        reply = bot_move(board, state["model"], state["device"], state["sims"])
        moves.append(describe_move(board, reply))
        board.push(reply)

        score = analyse(board)
        cp_after_bot = score.pov(bot_mover).score(mate_score=10000)
        bot_quality, bot_cp_loss = classify_move(cp_before_bot, cp_after_bot)
        moves[1]["quality"] = bot_quality
        moves[1]["cp_loss"] = bot_cp_loss
        state["move_qualities"].append({"quality": bot_quality, "cp_loss": bot_cp_loss})

    state["last_score"] = score
    return jsonify({**serialize(board, score), "moves": moves})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", nargs="?", default="/opt/ml/model/chessformer.safetensors"
    )
    parser.add_argument("--stockfish-path", default="/usr/games/stockfish")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--eval-depth", type=int, default=14)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    config = Config(stockfish_path=args.stockfish_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["board"] = chess.Board()
    state["model"] = load_model(args.checkpoint, device, config)
    state["device"] = device
    state["sims"] = args.sims
    state["eval_depth"] = args.eval_depth
    state["engine"] = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    state["move_qualities"] = []
    state["last_score"] = None

    try:
        app.run(host=args.host, port=args.port, threaded=False)
    finally:
        state["engine"].quit()


if __name__ == "__main__":
    main()
