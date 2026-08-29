import math
import time
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
        "raw_prior",
        "children",
        "visit_count",
        "value_sum",
        "virtual_loss",
        "expanded",
        "terminal",
        "terminal_value",
        "legal_moves",
    )

    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = None if parent is None else weakref.proxy(parent)
        self.move = move
        self.prior = prior

        self.raw_prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.expanded = False
        self.terminal = False
        self.terminal_value = 0.0
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

        exploration = c_puct * math.sqrt(max(1, self.visit_count + self.virtual_loss))
        best_score, best = float("-inf"), None
        for item in self.children.items():
            child = item[1]
            n = child.visit_count + child.virtual_loss
            q = 0.0 if n == 0 else (child.value_sum - child.virtual_loss) / n
            score = q + exploration * child.prior / (1 + n)
            if score > best_score:
                best_score, best = score, item
        return best if best is not None else next(iter(self.children.items()))

    def visit_distribution(self):
        total = sum(child.visit_count for child in self.children.values())
        if total == 0:
            return {move: 1.0 / len(self.children) for move in self.children}
        return {
            move: child.visit_count / total for move, child in self.children.items()
        }


def terminal_value(board, legal_moves=None):
    if legal_moves is None:
        legal_moves = list(board.legal_moves)
    if not legal_moves:
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return 0.0
    return 0.0 if board.is_repetition(3) else None


def game_result(board, legal_moves=None):
    value = terminal_value(board, legal_moves)
    if value is None:
        return False, None
    return True, (not board.turn) if value < 0 else None


def game_over(board, legal_moves=None):
    return terminal_value(board, legal_moves) is not None


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

    moves, slots, dests = [], [], []
    for (frm, to), move in move_map.items():
        slot = slot_of_square.get(frm)
        if slot is not None:
            moves.append(move)
            slots.append(slot)
            dests.append(to)

    if not moves:
        return

    logits = heatmap_row[slots, dests]

    if not np.isfinite(logits).all():
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
def _evaluate_boards(boards, model, device, legal_moves=None):
    if legal_moves is None:
        legal_moves = [list(b.legal_moves) for b in boards]
    board_inputs = torch.tensor(
        [board_to_input(b) for b in boards],
        dtype=torch.long,
        device=device,
    )
    heatmap, value, _ = model(board_inputs)
    piece_squares, piece_mask = piece_gather(board_inputs[:, :BOARD_SQUARES])
    return (
        heatmap.float().cpu().numpy(),
        value.float().cpu().tolist(),
        piece_squares.cpu().numpy(),
        piece_mask.cpu().numpy(),
        legal_moves,
    )


def _evaluate_boards_capped(boards, model, device, max_batch_size=None, moves=None):
    if not max_batch_size or len(boards) <= max_batch_size:
        return _evaluate_boards(boards, model, device, moves)

    heatmaps, values, piece_squares, piece_masks, legal_moves = [], [], [], [], []
    for start in range(0, len(boards), max_batch_size):
        chunk = boards[start : start + max_batch_size]
        chunk_moves = None if moves is None else moves[start : start + max_batch_size]
        hm, v, ps, pm, lm = _evaluate_boards(chunk, model, device, chunk_moves)
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
    deadline=None,
):
    live_roots = [
        root
        for root in roots
        if not root.terminal
        and terminal_value(root.ensure_board(), root.legal_moves) is None
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
        for root in live_roots:
            add_root_dirichlet_noise(root, root_dirichlet_alpha, root_noise_frac)

    effective_wave = (
        sims_per_wave
        if not target_batch_size or not live_roots
        else min(
            max(sims_per_wave, target_batch_size // len(live_roots)),
            max(sims_per_wave, num_simulations // 8),
        )
    )

    wave_cap = effective_wave if deadline is None else sims_per_wave
    sim_cost = None
    remaining = num_simulations
    while remaining > 0 and live_roots:
        wave = min(wave_cap, remaining)
        remaining -= wave
        wave_started = time.monotonic() if deadline is not None else 0.0

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
            board = leaf.ensure_board()
            leaf.legal_moves = list(board.legal_moves)
            tv = terminal_value(board, leaf.legal_moves)
            if tv is not None:
                leaf.terminal = True
                leaf.terminal_value = tv
            else:
                pending.append(leaf)

        if pending:
            heatmaps, values, piece_squares, piece_masks, _ = _evaluate_boards_capped(
                [leaf.board for leaf in pending],
                model,
                device,
                max_batch_size,
                [leaf.legal_moves for leaf in pending],
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
                value = leaf.terminal_value
            elif not math.isfinite(value):
                _warn_once(
                    f"non-finite leaf value ({value}) at fen={leaf.board.fen()!r}; using 0.0",
                    leaf.board.fen(),
                )
                value = 0.0
            _backup(path, value)

        if deadline is not None:
            now = time.monotonic()
            cost = max(now - wave_started, 1e-9) / wave
            sim_cost = cost if sim_cost is None else max(cost, 0.5 * sim_cost + cost)
            left = deadline - now
            if left <= sim_cost * sims_per_wave:
                break
            wave_cap = int(
                max(sims_per_wave, min(effective_wave, 0.35 * left / sim_cost))
            )

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


def improved_policy_pairs(root, mover, beta):
    if beta <= 0.0 or not root.children:
        return visit_policy_pairs(root, mover)
    moves = list(root.children)
    children = [root.children[m] for m in moves]
    visits = np.array([c.visit_count for c in children], dtype=np.float64)
    if visits.max() <= 0:
        return visit_policy_pairs(root, mover)

    root_q = -root.value_sum / root.visit_count if root.visit_count else 0.0
    q = np.array(
        [c.value_sum / c.visit_count if c.visit_count else root_q for c in children],
        dtype=np.float64,
    )
    span = q.max() - q.min()
    q = (q - q.min()) / span if span > 0 else np.zeros_like(q)

    prior = np.array([c.raw_prior for c in children], dtype=np.float64)
    prior = prior / max(prior.sum(), 1e-9)
    logits = np.log(np.clip(prior, 1e-9, None))
    logits += beta * (50.0 + visits.max()) / 50.0 * q
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    pairs = {}
    for move, prob in zip(moves, probs):
        key = (
            canon_square(move.from_square, mover),
            canon_square(move.to_square, mover),
        )
        pairs[key] = pairs.get(key, 0.0) + float(prob)
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
    target_batch_size=None,
    max_batch_size=None,
    deadline=None,
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
        target_batch_size=target_batch_size,
        max_batch_size=max_batch_size,
        deadline=deadline,
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


def mcts_move_with_visits(
    board,
    model,
    device,
    config,
    temperature=0.0,
    add_root_noise=False,
    num_simulations=None,
    deadline=None,
):
    moves, roots = mcts_policy_step(
        [board],
        model,
        device,
        num_simulations=(
            config.inference_mcts_simulations
            if num_simulations is None
            else num_simulations
        ),
        sims_per_wave=config.mcts_sims_per_wave,
        c_puct=config.mcts_c_puct,
        temperature=temperature,
        add_root_noise=add_root_noise,
        target_batch_size=config.mcts_target_batch_size,
        max_batch_size=config.mcts_max_batch_size,
        deadline=deadline,
    )
    move = moves[0]
    if move is None:
        move = next(iter(board.legal_moves))
    return move, (roots[0].visit_count if roots else 0)


def mcts_move(board, model, device, config, temperature=0.0, add_root_noise=False):
    move, _ = mcts_move_with_visits(
        board,
        model,
        device,
        config,
        temperature=temperature,
        add_root_noise=add_root_noise,
    )
    return move
