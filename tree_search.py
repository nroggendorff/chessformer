import math
import weakref

import numpy as np
import torch

from encoding import BOARD_SQUARES, board_square_tokens, board_to_input, child_board


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
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _softmax(logits):
    arr = np.asarray(logits, dtype=np.float64)
    exp = np.exp(arr - arr.max())
    return exp / exp.sum()


def expand_node(node, log_probs_row):
    moves = node.legal_moves
    if not moves:
        return
    child_tokens = np.array(
        [board_square_tokens(child_board(node.board, move)) for move in moves],
        dtype=np.int64,
    )
    scores = log_probs_row[np.arange(BOARD_SQUARES)[None, :], child_tokens].sum(axis=-1)
    for move, prior in zip(moves, _softmax(scores)):
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
    board_logits, value = model(board_inputs)
    log_probs = torch.log_softmax(board_logits, dim=-1)
    return (
        log_probs.float().cpu().numpy(),
        value.float().cpu().tolist(),
        legal_moves,
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
    target_batch_size=None,
):
    roots = [MCTSNode(board.copy()) for board in boards]

    live_roots = [root for root in roots if terminal_value(root.board) is None]
    if live_roots:
        log_probs, _, legal_moves = _evaluate_boards(
            [root.board for root in live_roots], model, device
        )
        for root, lp_row, lm in zip(live_roots, log_probs, legal_moves):
            root.legal_moves = lm
            expand_node(root, lp_row)
            if add_root_noise:
                add_root_dirichlet_noise(root, root_dirichlet_alpha, root_noise_frac)

    effective_wave = (
        sims_per_wave
        if not target_batch_size or not live_roots
        else max(sims_per_wave, target_batch_size // len(live_roots))
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
            log_probs, values, legal_moves = _evaluate_boards(
                [leaf.board for leaf in pending], model, device
            )
            leaf_values = {}
            for leaf, lp_row, lm, v in zip(pending, log_probs, legal_moves, values):
                leaf.legal_moves = lm
                expand_node(leaf, lp_row)
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
