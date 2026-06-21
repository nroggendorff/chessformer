import argparse

import chess
import chess.engine
import chess.svg
import torch
from flask import Flask, jsonify, request
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
        return ("Black" if board.turn else "White") + " wins by Checkmate"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Draw (Insufficient Material)"
    if board.can_claim_draw():
        return "Draw Available"
    return ("White" if board.turn == chess.WHITE else "Black") + " to Move"


def serialize(board):
    pieces = []

    for sq, piece in board.piece_map().items():
        pieces.append(
            {
                "square": chess.square_name(sq),
                "symbol": piece.symbol(),
            }
        )

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
        "eval": evaluate_position(board),
        "check_square": check_square,
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
                jsonify({"error": "Illegal or unparseable move", **serialize(board)}),
                400,
            )

    board.push(move)
    if not board.is_game_over(claim_draw=True):
        board.push(bot_move(board, state["model"], state["device"], state["sims"]))

    return jsonify(serialize(board))


INDEX_HTML = r"""
<head>
  <meta charset="utf-8" />
  <title>Play ChessNet</title>

  <style>
    :root {
      --bg-color: #222222;
      --panel-bg: #333333;
      --text-main: #eeeeee;
      --text-light: #ffffff;
      --accent: #007bff;
      --accent-hover: #0056b3;
      --border-color: #444444;
      --board-size: 480px;
    }

    body {
      background: var(--bg-color);
      color: var(--text-main);
      font-family:
        -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
        sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
    }

    #layout {
      display: flex;
      gap: 24px;
      align-items: stretch;
    }

    #evalbar-container {
      width: 24px;
      background: var(--panel-bg);
      border-radius: 4px;
      overflow: hidden;
      position: relative;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    #evalfill {
      position: absolute;
      bottom: 0;
      width: 100%;
      background: #cccccc;
      transition: height 0.3s ease-in-out;
    }

    #evaltext {
      position: absolute;
      top: 6px;
      width: 100%;
      text-align: center;
      font-size: 10px;
      font-weight: bold;
      color: #222222;
      z-index: 2;
    }

    #board-container {
      position: relative;
      width: var(--board-size);
      height: var(--board-size);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
      border-radius: 4px;
      overflow: hidden;
    }

    #board-bg {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
    }

    #board-bg svg {
      border-radius: 4px;
      width: 100%;
      height: 100%;
    }

    #board-pieces {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }

    .piece {
      position: absolute;
      width: 60px;
      height: 60px;
      cursor: grab;
      user-select: none;
      z-index: 10;
      pointer-events: auto;
      transition:
        left 0.25s ease-in-out,
        top 0.25s ease-in-out,
        opacity 0.2s ease-out;
    }

    .piece.dragging {
      z-index: 100;
      cursor: grabbing;
      transition: none;
      transform: scale(1.15);
      filter: drop-shadow(0 8px 12px rgba(0, 0, 0, 0.6));
    }

    .piece.in-check {
      background: radial-gradient(
        circle,
        rgba(235, 64, 52, 1) 0%,
        rgba(235, 64, 52, 0.4) 60%,
        rgba(0, 0, 0, 0) 70%
      );
      border-radius: 50%;
      animation: check-pulse 1.2s infinite alternate ease-in-out;
    }

    @keyframes check-pulse {
      from {
        filter: drop-shadow(0 0 4px rgba(235, 64, 52, 0.6));
        background: radial-gradient(
          circle,
          rgba(235, 64, 52, 0.95) 0%,
          rgba(235, 64, 52, 0.35) 60%,
          rgba(0, 0, 0, 0) 70%
        );
      }
      to {
        filter: drop-shadow(0 0 14px rgba(235, 64, 52, 0.95));
        background: radial-gradient(
          circle,
          rgba(235, 64, 52, 1) 0%,
          rgba(235, 64, 52, 0.55) 60%,
          rgba(0, 0, 0, 0) 70%
        );
      }
    }

    .coord {
      position: absolute;
      font-size: 11px;
      font-weight: bold;
      pointer-events: none;
      user-select: none;
      z-index: 5;
    }

    .coord-file {
      bottom: 3px;
      right: 5px;
    }

    .coord-rank {
      top: 4px;
      left: 4px;
    }

    #promotion-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.75);
      display: none;
      justify-content: center;
      align-items: center;
      z-index: 1000;
    }

    .promotion-dialog {
      background: var(--panel-bg);
      border: 2px solid var(--border-color);
      border-radius: 8px;
      padding: 24px;
      text-align: center;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      width: 250px;
    }

    .promotion-dialog h3 {
      margin: 0 0 16px 0;
      color: var(--text-light);
      font-size: 18px;
    }

    .promotion-options {
      display: flex;
      gap: 12px;
      justify-content: center;
      margin-bottom: 20px;
    }

    .promotion-options button {
      width: 50px;
      height: 50px;
      padding: 4px;
      background: transparent;
      border-radius: 4px;
      border: 1px solid var(--border-color);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition:
        background 0.2s,
        transform 0.1s;
    }

    .promotion-options button:hover {
      background: var(--border-color);
      transform: scale(1.1);
    }

    .promotion-options svg {
      width: 100%;
      height: 100%;
    }

    .cancel-btn {
      background: #d9534f !important;
      color: white !important;
      font-size: 14px !important;
      padding: 10px 20px !important;
      width: auto !important;
      display: inline-block;
    }

    .cancel-btn:hover {
      background: #c9302c !important;
    }

    #sidebar {
      display: flex;
      flex-direction: column;
      justify-content: center;
      width: 320px;
    }

    .panel {
      background: var(--panel-bg);
      border-radius: 4px;
      padding: 24px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    #status {
      font-size: 22px;
      font-weight: bold;
      color: var(--text-light);
      margin: 0 0 8px 0;
    }

    #error {
      color: #d9534f;
      min-height: 20px;
      font-size: 14px;
      font-weight: bold;
      margin-bottom: 20px;
    }

    .btn-row {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    button {
      padding: 14px;
      background: var(--border-color);
      color: var(--text-main);
      border: none;
      border-radius: 4px;
      font-size: 16px;
      font-weight: bold;
      cursor: pointer;
      transition:
        background 0.2s,
        color 0.2s;
      width: 100%;
    }

    button:hover {
      background: #555555;
      color: var(--text-light);
    }

    button.primary {
      background: var(--accent);
      color: var(--text-light);
    }

    button.primary:hover {
      background: var(--accent-hover);
    }
  </style>
</head>

<body>
  <div id="layout">
    <div id="evalbar-container">
      <div id="evaltext">0.00</div>
      <div id="evalfill" style="height: 50%"></div>
    </div>

    <div id="board-container">
      <div id="board-bg"></div>
      <div id="board-pieces"></div>

      <div id="promotion-overlay">
        <div class="promotion-dialog">
          <h3>Promote Pawn</h3>
          <div class="promotion-options">
            <button data-piece="q" title="Queen">
              <svg viewBox="0 0 45 45">
                <use id="promo-q-use" href=""></use>
              </svg>
            </button>
            <button data-piece="r" title="Rook">
              <svg viewBox="0 0 45 45">
                <use id="promo-r-use" href=""></use>
              </svg>
            </button>
            <button data-piece="b" title="Bishop">
              <svg viewBox="0 0 45 45">
                <use id="promo-b-use" href=""></use>
              </svg>
            </button>
            <button data-piece="n" title="Knight">
              <svg viewBox="0 0 45 45">
                <use id="promo-n-use" href=""></use>
              </svg>
            </button>
          </div>
          <button type="button" id="promotion-cancel" class="cancel-btn">
            Cancel
          </button>
        </div>
      </div>
    </div>

    <div id="sidebar">
      <div class="panel">
        <h2 id="status">Loading...</h2>
        <div id="error"></div>

        <div class="btn-row">
          <button type="button" id="newgame" class="primary">New Game</button>
          <button type="button" id="undo">Undo Move</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const SYMBOL_TO_ID = {
      P: "white-pawn",
      N: "white-knight",
      B: "white-bishop",
      R: "white-rook",
      Q: "white-queen",
      K: "white-king",
      p: "black-pawn",
      n: "black-knight",
      b: "black-bishop",
      r: "black-rook",
      q: "black-queen",
      k: "black-king",
    };

    const LIGHT_COLOR = "#f0d9b5";
    const DARK_COLOR = "#b58863";

    let pendingPromotion = null;

    function squarePos(square) {
      const file = "abcdefgh".indexOf(square[0]);
      const rank = parseInt(square[1]);
      return {
        left: file * 60,
        top: (8 - rank) * 60,
      };
    }

    function squareFromPoint(x, y) {
      const rect = document
        .getElementById("board-container")
        .getBoundingClientRect();
      const bx = x - rect.left;
      const by = y - rect.top;

      const file = Math.floor(bx / 60);
      const rank = 7 - Math.floor(by / 60);

      if (file < 0 || file > 7 || rank < 0 || rank > 7) {
        return null;
      }

      return "abcdefgh"[file] + (rank + 1);
    }

    function initCoordinates() {
      const boardContainer = document.getElementById("board-container");

      for (let r = 1; r <= 8; r++) {
        const div = document.createElement("div");
        div.className = "coord coord-rank";
        div.textContent = r;

        const isLightSquare = r % 2 === 0;
        div.style.color = isLightSquare ? DARK_COLOR : LIGHT_COLOR;

        const pos = squarePos("a" + r);
        div.style.top = pos.top + "px";
        boardContainer.appendChild(div);
      }

      const files = "abcdefgh";
      for (let f = 0; f < 8; f++) {
        const div = document.createElement("div");
        div.className = "coord coord-file";
        div.textContent = files[f];

        const isLightSquare = f % 2 !== 0;
        div.style.color = isLightSquare ? DARK_COLOR : LIGHT_COLOR;

        const pos = squarePos(files[f] + "1");
        div.style.left = pos.left + "px";
        boardContainer.appendChild(div);
      }
    }

    function createPieceElement(symbol, square) {
      const div = document.createElement("div");
      div.className = "piece";
      div.dataset.symbol = symbol;
      div.dataset.square = square;

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 45 45");
      svg.style.width = "100%";
      svg.style.height = "100%";

      const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
      use.setAttribute("href", "#" + SYMBOL_TO_ID[symbol]);

      svg.appendChild(use);
      div.appendChild(svg);

      makeDraggable(div);
      document.getElementById("board-pieces").appendChild(div);
      return div;
    }

    function renderPieces(newPieces, checkSquare) {
      const piecesContainer = document.getElementById("board-pieces");
      const currentPieceElements = Array.from(
        piecesContainer.querySelectorAll(".piece"),
      );

      const matchedCurrent = new Set();
      const matchedNew = new Set();

      function updateCheckHighlight(elem, symbol, sq) {
        if (sq === checkSquare && (symbol === "K" || symbol === "k")) {
          elem.classList.add("in-check");
        } else {
          elem.classList.remove("in-check");
        }
      }

      newPieces.forEach((np, i) => {
        const matchIdx = currentPieceElements.findIndex(
          (cp, j) =>
            !matchedCurrent.has(j) &&
            cp.dataset.symbol === np.symbol &&
            cp.dataset.square === np.square,
        );

        if (matchIdx !== -1) {
          matchedCurrent.add(matchIdx);
          matchedNew.add(i);
          const piece = currentPieceElements[matchIdx];
          updateCheckHighlight(piece, np.symbol, np.square);
        }
      });

      newPieces.forEach((np, i) => {
        if (matchedNew.has(i)) return;

        const matchIdx = currentPieceElements.findIndex(
          (cp, j) => !matchedCurrent.has(j) && cp.dataset.symbol === np.symbol,
        );

        if (matchIdx !== -1) {
          matchedCurrent.add(matchIdx);
          matchedNew.add(i);

          const piece = currentPieceElements[matchIdx];
          piece.dataset.square = np.square;
          const pos = squarePos(np.square);
          piece.style.left = pos.left + "px";
          piece.style.top = pos.top + "px";
          updateCheckHighlight(piece, np.symbol, np.square);
        }
      });

      newPieces.forEach((np, i) => {
        if (matchedNew.has(i)) return;

        const div = createPieceElement(np.symbol, np.square);
        const pos = squarePos(np.square);
        div.style.left = pos.left + "px";
        div.style.top = pos.top + "px";
        updateCheckHighlight(div, np.symbol, np.square);
      });

      currentPieceElements.forEach((cp, j) => {
        if (!matchedCurrent.has(j)) {
          cp.style.opacity = "0";
          setTimeout(() => cp.remove(), 200);
        }
      });
    }

    function showPromotionModal(color) {
      const prefix = color === "white" ? "white-" : "black-";
      document
        .getElementById("promo-q-use")
        .setAttribute("href", "#" + prefix + "queen");
      document
        .getElementById("promo-r-use")
        .setAttribute("href", "#" + prefix + "rook");
      document
        .getElementById("promo-b-use")
        .setAttribute("href", "#" + prefix + "bishop");
      document
        .getElementById("promo-n-use")
        .setAttribute("href", "#" + prefix + "knight");
      document.getElementById("promotion-overlay").style.display = "flex";
    }

    function hidePromotionModal() {
      document.getElementById("promotion-overlay").style.display = "none";
      pendingPromotion = null;
    }

    async function handlePromotionSelection(pieceChar) {
      if (!pendingPromotion) return;
      const { startSquare, target, piece } = pendingPromotion;
      const fullMove = startSquare + target + pieceChar;

      const targetPos = squarePos(target);
      piece.style.left = targetPos.left + "px";
      piece.style.top = targetPos.top + "px";
      piece.dataset.square = target;
      piece.classList.remove("dragging");

      hidePromotionModal();

      const response = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ move: fullMove }),
      });

      const data = await response.json();

      if (!response.ok) {
        document.getElementById("error").textContent =
          data.error || "Invalid move";
      }
      render(data);
    }

    function makeDraggable(piece) {
      piece.addEventListener("mousedown", (e) => {
        if (pendingPromotion) return;

        const startSquare = piece.dataset.square;
        const rect = piece.getBoundingClientRect();

        const ox = e.clientX - rect.left;
        const oy = e.clientY - rect.top;

        piece.classList.add("dragging");

        function move(ev) {
          const board = document
            .getElementById("board-container")
            .getBoundingClientRect();
          piece.style.left = ev.clientX - board.left - ox + "px";
          piece.style.top = ev.clientY - board.top - oy + "px";
        }

        async function up(ev) {
          document.removeEventListener("mousemove", move);
          document.removeEventListener("mouseup", up);

          const target = squareFromPoint(ev.clientX, ev.clientY);

          if (!target || target === startSquare) {
            piece.classList.remove("dragging");
            const pos = squarePos(startSquare);
            piece.style.left = pos.left + "px";
            piece.style.top = pos.top + "px";
            return;
          }

          const isWhitePawnPromo =
            piece.dataset.symbol === "P" &&
            startSquare[1] === "7" &&
            target[1] === "8";
          const isBlackPawnPromo =
            piece.dataset.symbol === "p" &&
            startSquare[1] === "2" &&
            target[1] === "1";

          if (isWhitePawnPromo || isBlackPawnPromo) {
            const targetPos = squarePos(target);
            piece.style.left = targetPos.left + "px";
            piece.style.top = targetPos.top + "px";
            piece.classList.remove("dragging");

            pendingPromotion = { startSquare, target, piece };
            showPromotionModal(isWhitePawnPromo ? "white" : "black");
            return;
          }

          const targetPos = squarePos(target);
          piece.style.left = targetPos.left + "px";
          piece.style.top = targetPos.top + "px";
          piece.dataset.square = target;
          piece.classList.remove("dragging");

          const response = await fetch("/api/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ move: startSquare + target }),
          });

          const data = await response.json();

          if (!response.ok) {
            document.getElementById("error").textContent =
              data.error || "Invalid move";
          }

          render(data);
        }

        document.addEventListener("mousemove", move);
        document.addEventListener("mouseup", up);
      });
    }

    function render(data) {
      const tempDiv = document.createElement("div");
      tempDiv.innerHTML = data.svg;
      const pieceUses = tempDiv.querySelectorAll("use");
      pieceUses.forEach((u) => {
        const href = u.getAttribute("href") || u.getAttribute("xlink:href");
        if (
          href &&
          (href.startsWith("#white-") || href.startsWith("#black-"))
        ) {
          u.remove();
        }
      });

      document.getElementById("board-bg").innerHTML = tempDiv.innerHTML;

      renderPieces(data.pieces, data.check_square);

      document.getElementById("status").textContent = data.status;
      document.getElementById("error").textContent = data.error || "";

      const pct =
        data.eval.mate !== undefined
          ? data.eval.mate > 0
            ? 100
            : 0
          : 50 + 50 * Math.tanh(data.eval.cp / 400);

      document.getElementById("evalfill").style.height = pct + "%";

      const evalTextEl = document.getElementById("evaltext");
      evalTextEl.textContent =
        data.eval.mate !== undefined
          ? "M" + Math.abs(data.eval.mate)
          : (data.eval.cp / 100).toFixed(2);

      evalTextEl.style.color = pct > 80 ? "#222222" : "#cccccc";
    }

    async function refresh() {
      render(await (await fetch("/api/state")).json());
    }

    document.getElementById("newgame").addEventListener("click", async () => {
      hidePromotionModal();
      render(await (await fetch("/api/new", { method: "POST" })).json());
    });

    document.getElementById("undo").addEventListener("click", async () => {
      hidePromotionModal();
      render(await (await fetch("/api/undo", { method: "POST" })).json());
    });

    document
      .getElementById("promotion-cancel")
      .addEventListener("click", () => {
        if (pendingPromotion) {
          const { startSquare, piece } = pendingPromotion;
          piece.classList.remove("dragging");
          const pos = squarePos(startSquare);
          piece.style.left = pos.left + "px";
          piece.style.top = pos.top + "px";
        }
        hidePromotionModal();
      });

    document.querySelectorAll(".promotion-options button").forEach((btn) => {
      btn.addEventListener("click", () => {
        handlePromotionSelection(btn.dataset.piece);
      });
    });

    initCoordinates();
    refresh();
  </script>
</body>
"""


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

    try:
        app.run(host=args.host, port=args.port, threaded=False)
    finally:
        state["engine"].quit()


if __name__ == "__main__":
    main()
