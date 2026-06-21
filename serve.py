import argparse

import chess
import chess.engine
import chess.svg
import torch
from flask import Flask, jsonify, request
from safetensors.torch import load_file

from mcts import MCTSNode, batched_mcts_sim, root_converged, root_policy_from_visits
from model import ChessNet

app = Flask(__name__)
state = {}


def load_model(checkpoint_path, device):
    model = ChessNet().to(device)
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


def evaluate_position(board):
    info = state["engine"].analyse(board, chess.engine.Limit(depth=state["eval_depth"]))
    score = info["score"].pov(chess.WHITE)
    return (
        {"mate": score.mate()}
        if score.is_mate()
        else {"cp": score.score(mate_score=10000)}
    )


def game_status(board):
    if board.is_checkmate():
        return ("Black" if board.turn else "White") + " wins by checkmate"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Draw, insufficient material"
    if board.can_claim_draw():
        return "Draw available"
    return ("White" if board.turn == chess.WHITE else "Black") + " to move"


def serialize(board):
    return {
        "svg": chess.svg.board(
            board, lastmove=board.peek() if board.move_stack else None, size=480
        ),
        "fen": board.fen(),
        "status": game_status(board),
        "game_over": board.is_game_over(claim_draw=True),
        "eval": evaluate_position(board),
    }


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/state")
def api_state():
    return jsonify(serialize(state["board"]))


@app.route("/api/new", methods=["POST"])
def api_new():
    state["board"] = chess.Board()
    return jsonify(serialize(state["board"]))


@app.route("/api/undo", methods=["POST"])
def api_undo():
    board = state["board"]
    for _ in range(2):
        if board.move_stack:
            board.pop()
    return jsonify(serialize(board))


@app.route("/api/move", methods=["POST"])
def api_move():
    board = state["board"]
    if board.is_game_over(claim_draw=True):
        return jsonify(serialize(board))

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
                jsonify({"error": "illegal or unparseable move", **serialize(board)}),
                400,
            )

    board.push(move)
    if not board.is_game_over(claim_draw=True):
        board.push(bot_move(board, state["model"], state["device"], state["sims"]))

    return jsonify(serialize(board))


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Play ChessNet</title>
<style>
body { background: #1e1e1e; color: #ddd; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; padding-top: 30px; }
#layout { display: flex; gap: 16px; align-items: flex-start; }
#evalbar { width: 30px; height: 480px; background: #111; position: relative; border: 1px solid #444; }
#evalfill { position: absolute; bottom: 0; width: 100%; background: #eee; transition: height 0.3s; }
#evaltext { position: absolute; top: -22px; width: 100%; text-align: center; font-size: 12px; }
#board svg { display: block; }
#controls { margin-top: 16px; display: flex; gap: 8px; }
#status { margin-top: 10px; font-size: 14px; }
input { padding: 6px; font-size: 14px; }
button { padding: 6px 12px; font-size: 14px; cursor: pointer; }
#error { color: #e66; margin-top: 6px; font-size: 13px; min-height: 18px; }
</style>
</head>
<body>
<div id="layout">
<div id="evalbar"><div id="evaltext">0.00</div><div id="evalfill" style="height:50%"></div></div>
<div id="board"></div>
</div>
<div id="status"></div>
<div id="error"></div>
<form id="moveform" autocomplete="off">
<div id="controls">
<input id="moveinput" placeholder="e2e4 or Nf3" autofocus>
<button type="submit">Move</button>
<button type="button" id="undo">Undo</button>
<button type="button" id="newgame">New Game</button>
</div>
</form>
<script>
function render(data) {
  document.getElementById("board").innerHTML = data.svg;
  document.getElementById("status").textContent = data.status;
  document.getElementById("error").textContent = data.error || "";
  const pct = data.eval.mate !== undefined
    ? (data.eval.mate > 0 ? 100 : 0)
    : 50 + 50 * Math.tanh(data.eval.cp / 400);
  document.getElementById("evalfill").style.height = pct + "%";
  document.getElementById("evaltext").textContent = data.eval.mate !== undefined
    ? "M" + Math.abs(data.eval.mate)
    : (data.eval.cp / 100).toFixed(2);
}

async function refresh() {
  render(await (await fetch("/api/state")).json());
}

document.getElementById("moveform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("moveinput");
  const data = await (await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ move: input.value }),
  })).json();
  input.value = "";
  render(data);
});

document.getElementById("newgame").addEventListener("click", async () => {
  render(await (await fetch("/api/new", { method: "POST" })).json());
});

document.getElementById("undo").addEventListener("click", async () => {
  render(await (await fetch("/api/undo", { method: "POST" })).json());
});

refresh();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--stockfish-path", default="/usr/games/stockfish")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--eval-depth", type=int, default=14)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["board"] = chess.Board()
    state["model"] = load_model(args.checkpoint, device)
    state["device"] = device
    state["sims"] = args.sims
    state["eval_depth"] = args.eval_depth
    state["engine"] = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

    try:
        app.run(host=args.host, port=args.port, threaded=False)
    finally:
        state["engine"].quit()


if __name__ == "__main__":
    main()
