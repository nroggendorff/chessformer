import math

import numpy as np
import torch

from encoding import (
    BOARD_SQUARES,
    board_to_input,
    canonical_square,
    legal_moves_by_square_pair,
)
from model import piece_gather
from policy import resolve_promotions


class MCTSNode:
    __slots__ = (
        "board",
        "parent",
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "virtual_loss",
        "expanded",
        "terminal",
    )

    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.expanded = False
        self.terminal = False

    def puct_score(self, c_puct, parent_visits):
        n = self.visit_count + self.virtual_loss
        q = 0.0 if n == 0 else (self.value_sum - self.virtual_loss) / n
        return q + c_puct * self.prior * math.sqrt(parent_visits) / (1 + n)

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
    move_map = legal_moves_by_square_pair(node.board)
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

    for move, prior in zip(moves, _softmax(logits)):
        child_board = node.board.copy()
        child_board.push(move)
        node.children[move] = MCTSNode(child_board, parent=node, prior=float(prior))
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
    sign = 1.0
    for node in reversed(path):
        node.virtual_loss -= 1
        node.visit_count += 1
        node.value_sum += sign * value
        sign = -sign


@torch.inference_mode()
def _evaluate_boards(boards, model, device):
    board_inputs = torch.tensor(
        [board_to_input(b) for b in boards], dtype=torch.long, device=device
    )
    heatmap, value = model(board_inputs)
    piece_squares, piece_mask = piece_gather(board_inputs[:, :BOARD_SQUARES])
    return (
        heatmap.float().cpu().numpy(),
        value.float().cpu().tolist(),
        piece_squares.cpu().numpy(),
        piece_mask.cpu().numpy(),
    )


def run_mcts(
    boards,
    model,
    device,
    num_simulations=200,
    sims_per_wave=8,
    c_puct=1.5,
    add_root_noise=False,
    root_dirichlet_alpha=0.3,
    root_noise_frac=0.25,
):
    roots = [MCTSNode(board.copy()) for board in boards]

    live_roots = [root for root in roots if terminal_value(root.board) is None]
    if live_roots:
        heatmaps, _, piece_squares, piece_masks = _evaluate_boards(
            [root.board for root in live_roots], model, device
        )
        for root, hm_row, ps_row, pm_row in zip(
            live_roots, heatmaps, piece_squares, piece_masks
        ):
            expand_node(root, hm_row, ps_row, pm_row)
            if add_root_noise:
                add_root_dirichlet_noise(root, root_dirichlet_alpha, root_noise_frac)

    remaining = num_simulations
    while remaining > 0 and live_roots:
        wave = min(sims_per_wave, remaining)
        remaining -= wave

        paths = [
            (root, path)
            for root in live_roots
            if root.children
            for _ in range(wave)
            for path in (_select_leaf(root, c_puct),)
        ]
        for _, path in paths:
            for node in path:
                node.virtual_loss += 1

        pending, seen = [], set()
        for _, path in paths:
            leaf = path[-1]
            if id(leaf) in seen or leaf.expanded or leaf.terminal:
                continue
            seen.add(id(leaf))
            tv = terminal_value(leaf.board)
            if tv is not None:
                leaf.terminal = True
            else:
                pending.append(leaf)

        if pending:
            heatmaps, values, piece_squares, piece_masks = _evaluate_boards(
                [leaf.board for leaf in pending], model, device
            )
            leaf_values = {}
            for leaf, hm_row, ps_row, pm_row, v in zip(
                pending, heatmaps, piece_squares, piece_masks, values
            ):
                expand_node(leaf, hm_row, ps_row, pm_row)
                leaf_values[id(leaf)] = v
        else:
            leaf_values = {}

        for _, path in paths:
            leaf = path[-1]
            value = leaf_values.get(id(leaf))
            if value is None:
                value = terminal_value(leaf.board) or 0.0
            _backup(path, value)

    return roots


def visit_policy_pairs(root):
    pairs = {}
    for move, prob in root.visit_distribution().items():
        key = (
            canonical_square(move.from_square, root.board),
            canonical_square(move.to_square, root.board),
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
        boards,
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
