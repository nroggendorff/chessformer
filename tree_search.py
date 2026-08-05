import math
import weakref

import numpy as np
import torch

from encoding import (
    BOARD_SQUARES,
    board_to_input,
    canon_square,
    legal_moves_by_square_pair,
)
from model import piece_gather
from policy import resolve_promotions

_warned_fens = set()


def _warn_once(message, fen):
    if fen not in _warned_fens:
        _warned_fens.add(fen)
        print(message)


class MCTSNode:
    __slots__ = (
        "__weakref__",
        "board",
        "parent",
        "move",
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "virtual_loss",
        "expanded",
        "terminal",
        "legal_moves",
    )

    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = None if parent is None else weakref.proxy(parent)
        self.move = move
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.expanded = False
        self.terminal = False
        self.legal_moves = None

    def ensure_board(self):
        if self.board is None:
            self.board = self.parent.board.copy()
            self.board.push(self.move)
        return self.board

    def puct_score(self, c_puct, parent_visits):
        n = self.visit_count + self.virtual_loss
        q = 0.0 if n == 0 else (self.value_sum - self.virtual_loss) / n
        score = q + c_puct * self.prior * math.sqrt(parent_visits) / (1 + n)
        return score if math.isfinite(score) else float("-inf")

    def select_child(self, c_puct):
        parent_visits = max(1, self.visit_count + self.virtual_loss)
        return max(
            self.children.items(),
            key=lambda kv: kv[1].puct_score(c_puct, parent_visits),
        )

    def visit_distribution(self):
        total = sum(child.visit_count for child in self.children.values())
        if total == 0:
            return {move: 1.0 / len(self.children) for move in self.children}
        return {
            move: child.visit_count / total for move, child in self.children.items()
        }


def terminal_value(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _softmax(logits):
    arr = np.asarray(logits, dtype=np.float64)
    exp = np.exp(arr - arr.max())
    return exp / exp.sum()


def expand_node(node, heatmap_row, piece_squares_row, piece_mask_row):
    move_map = legal_moves_by_square_pair(node.board, legal_moves=node.legal_moves)
    slot_of_square = {
        int(sq): slot
        for slot, sq in enumerate(piece_squares_row)
        if piece_mask_row[slot]
    }

    moves, logits = [], []
    for (frm, to), move in move_map.items():
        slot = slot_of_square.get(frm)
        if slot is not None:
            moves.append(move)
            logits.append(float(heatmap_row[slot, to]))

    if not moves:
        return

    if not all(math.isfinite(logit) for logit in logits):
        _warn_once(
            f"non-finite heatmap logits at fen={node.board.fen()!r}; using uniform prior",
            node.board.fen(),
        )
        priors = [1.0 / len(moves)] * len(moves)
    else:
        priors = _softmax(logits)

    for move, prior in zip(moves, priors):
        node.children[move] = MCTSNode(None, parent=node, move=move, prior=float(prior))
    node.expanded = True


def add_root_dirichlet_noise(root, alpha, frac):
    if not root.children:
        return
    noise = np.random.dirichlet([alpha] * len(root.children))
    for child, n in zip(root.children.values(), noise):
        child.prior = child.prior * (1 - frac) + float(n) * frac


def _select_leaf(root, c_puct):
    path = [root]
    node = root
    while node.expanded and not node.terminal and node.children:
        _, node = node.select_child(c_puct)
        path.append(node)
    return path


def _backup(path, value):
    sign = -1.0
    for node in reversed(path):
        node.virtual_loss -= 1
        node.visit_count += 1
        node.value_sum += sign * value
        sign = -sign


@torch.inference_mode()
def _evaluate_boards(boards, model, device):
    legal_moves = [list(b.legal_moves) for b in boards]
    board_inputs = torch.tensor(
        [board_to_input(b) for b in boards],
        dtype=torch.long,
        device=device,
    )
    heatmap, value = model(board_inputs)
    piece_squares, piece_mask = piece_gather(board_inputs[:, :BOARD_SQUARES])
    return (
        heatmap.float().cpu().numpy(),
        value.float().cpu().tolist(),
        piece_squares.cpu().numpy(),
        piece_mask.cpu().numpy(),
        legal_moves,
    )


def _evaluate_boards_capped(boards, model, device, max_batch_size=None):
    if not max_batch_size or len(boards) <= max_batch_size:
        return _evaluate_boards(boards, model, device)

    heatmaps, values, piece_squares, piece_masks, legal_moves = [], [], [], [], []
    for start in range(0, len(boards), max_batch_size):
        chunk = boards[start : start + max_batch_size]
        hm, v, ps, pm, lm = _evaluate_boards(chunk, model, device)
        heatmaps.append(hm)
        values.extend(v)
        piece_squares.append(ps)
        piece_masks.append(pm)
        legal_moves.extend(lm)
    return (
        np.concatenate(heatmaps),
        values,
        np.concatenate(piece_squares),
        np.concatenate(piece_masks),
        legal_moves,
    )


def run_mcts(
    roots,
    model,
    device,
    num_simulations=200,
    sims_per_wave=8,
    c_puct=1.5,
    add_root_noise=False,
    root_dirichlet_alpha=0.3,
    root_noise_frac=0.25,
    target_batch_size=None,
    max_batch_size=None,
):
    live_roots = [
        root
        for root in roots
        if not root.terminal and terminal_value(root.ensure_board()) is None
    ]
    fresh_roots = [root for root in live_roots if not root.expanded]
    if fresh_roots:
        heatmaps, _, piece_squares, piece_masks, legal_moves = _evaluate_boards_capped(
            [root.board for root in fresh_roots], model, device, max_batch_size
        )
        for root, hm_row, ps_row, pm_row, lm in zip(
            fresh_roots, heatmaps, piece_squares, piece_masks, legal_moves
        ):
            root.legal_moves = lm
            expand_node(root, hm_row, ps_row, pm_row)
            if add_root_noise:
                add_root_dirichlet_noise(root, root_dirichlet_alpha, root_noise_frac)

    effective_wave = (
        sims_per_wave
        if not target_batch_size or not live_roots
        else min(
            max(sims_per_wave, target_batch_size // len(live_roots)),
            max(sims_per_wave, num_simulations // 8),
        )
    )

    remaining = num_simulations
    while remaining > 0 and live_roots:
        wave = min(effective_wave, remaining)
        remaining -= wave

        paths = []
        for root in live_roots:
            if not root.children:
                continue
            for _ in range(wave):
                path = _select_leaf(root, c_puct)
                for node in path:
                    node.virtual_loss += 1
                paths.append((root, path))

        pending, seen = [], set()
        for _, path in paths:
            leaf = path[-1]
            if id(leaf) in seen or leaf.expanded or leaf.terminal:
                continue
            seen.add(id(leaf))
            tv = terminal_value(leaf.ensure_board())
            if tv is not None:
                leaf.terminal = True
            else:
                pending.append(leaf)

        if pending:
            heatmaps, values, piece_squares, piece_masks, legal_moves = (
                _evaluate_boards_capped(
                    [leaf.board for leaf in pending], model, device, max_batch_size
                )
            )
            leaf_values = {}
            for leaf, hm_row, ps_row, pm_row, lm, v in zip(
                pending, heatmaps, piece_squares, piece_masks, legal_moves, values
            ):
                leaf.legal_moves = lm
                expand_node(leaf, hm_row, ps_row, pm_row)
                leaf_values[id(leaf)] = v
        else:
            leaf_values = {}

        for _, path in paths:
            leaf = path[-1]
            value = leaf_values.get(id(leaf))
            if value is None:
                value = terminal_value(leaf.board) or 0.0
            elif not math.isfinite(value):
                _warn_once(
                    f"non-finite leaf value ({value}) at fen={leaf.board.fen()!r}; using 0.0",
                    leaf.board.fen(),
                )
                value = 0.0
            _backup(path, value)

    return roots


def visit_policy_pairs(root, mover):
    pairs = {}
    for move, prob in root.visit_distribution().items():
        key = (
            canon_square(move.from_square, mover),
            canon_square(move.to_square, mover),
        )
        pairs[key] = pairs.get(key, 0.0) + prob
    return pairs


def choose_move(root, temperature):
    if not root.children:
        return None
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    if temperature <= 0:
        return moves[int(visits.argmax())]
    weights = visits ** (1.0 / temperature)
    weights = weights / weights.sum()
    return moves[np.random.choice(len(moves), p=weights)]


def mcts_policy_step(
    boards,
    model,
    device,
    num_simulations=200,
    sims_per_wave=8,
    c_puct=1.5,
    temperature=0.0,
    add_root_noise=False,
    root_dirichlet_alpha=0.3,
    root_noise_frac=0.25,
):
    roots = run_mcts(
        [MCTSNode(board.copy()) for board in boards],
        model,
        device,
        num_simulations=num_simulations,
        sims_per_wave=sims_per_wave,
        c_puct=c_puct,
        add_root_noise=add_root_noise,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_noise_frac=root_noise_frac,
    )
    moves = [choose_move(root, temperature) for root in roots]
    live_idx = [i for i, m in enumerate(moves) if m is not None]
    if any(moves[i].promotion is not None for i in live_idx):
        resolved = resolve_promotions(
            [boards[i] for i in live_idx],
            [moves[i] for i in live_idx],
            model,
            device,
        )
        for i, move in zip(live_idx, resolved):
            moves[i] = move
    return moves, roots


def mcts_move(board, model, device, config, temperature=0.0, add_root_noise=False):
    moves, _ = mcts_policy_step(
        [board],
        model,
        device,
        num_simulations=config.inference_mcts_simulations,
        sims_per_wave=config.mcts_sims_per_wave,
        c_puct=config.mcts_c_puct,
        temperature=temperature,
        add_root_noise=add_root_noise,
    )
    move = moves[0]
    return move if move is not None else next(iter(board.legal_moves))
