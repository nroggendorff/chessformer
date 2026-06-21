import math
import random

import chess
import numpy as np
import torch

from encoding import board_to_tokens, canonical_square, legal_moves_by_square_pair


class MCTSNode:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "expanded",
        "move_map",
        "terminal_value",
    )

    def __init__(self, prior=1.0):
        self.prior = float(prior)
        self.visit_count, self.value_sum = 0, 0.0
        self.children = {}
        self.expanded = False
        self.move_map = None
        self.terminal_value = None

    @property
    def q(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def get_relative_value(board):
    out = board.outcome(claim_draw=True)
    return (
        0.0
        if not out or out.winner is None
        else (1.0 if out.winner == board.turn else -1.0)
    )


def select_child(node, cpuct):
    total_visits = math.sqrt(node.visit_count + 1e-8)
    return max(
        node.children.items(),
        key=lambda x: x[1].q
        + cpuct * x[1].prior * total_visits / (1.0 + x[1].visit_count),
    )


def add_root_noise(root, alpha=0.3, frac=0.25):
    if len(root.children) <= 1:
        return
    for child, n in zip(
        root.children.values(), np.random.dirichlet([alpha] * len(root.children))
    ):
        child.prior = (1 - frac) * child.prior + frac * n


@torch.inference_mode()
def batched_mcts_sim(roots, boards, model, device, cpuct=1.5, add_noise=False):
    batch_size = len(boards)
    search_paths, moves_pushed_list = [], []

    for root, board in zip(roots, boards):
        node, path, moves_pushed = root, [root], 0
        while node.expanded and node.children:
            move, child = select_child(node, cpuct)
            board.push(move)
            node = child
            path.append(node)
            moves_pushed += 1
        search_paths.append(path)
        moves_pushed_list.append(moves_pushed)

    eval_indices, eval_tokens, eval_move_maps = [], [], []
    leaf_values = [0.0] * batch_size

    for i, board in enumerate(boards):
        node = search_paths[i][-1]
        if node.terminal_value is not None:
            leaf_values[i] = node.terminal_value
            node.expanded = True
        elif board.is_game_over():
            node.terminal_value = get_relative_value(board)
            leaf_values[i] = node.terminal_value
            node.expanded = True
        else:
            eval_indices.append(i)
            eval_tokens.append(board_to_tokens(board))
            if node.move_map is None:
                node.move_map = legal_moves_by_square_pair(board)
            eval_move_maps.append(node.move_map)

    if eval_indices:
        b_tokens = torch.tensor(eval_tokens, dtype=torch.long, device=device)
        logits, val_preds = model(b_tokens)

        mask_cpu = torch.zeros((len(eval_indices), 64, 64), dtype=torch.bool)
        for idx, move_map in enumerate(eval_move_maps):
            for f, t in move_map:
                mask_cpu[idx, f, t] = True

        logits = logits.masked_fill(~mask_cpu.to(device, non_blocking=True), -1e4)
        probs = (
            torch.softmax(logits.view(len(eval_indices), -1), dim=-1)
            .view(len(eval_indices), 64, 64)
            .cpu()
        )
        val_preds = val_preds.cpu().tolist()

        for idx, original_i in enumerate(eval_indices):
            node, v, move_map = (
                search_paths[original_i][-1],
                val_preds[idx],
                eval_move_maps[idx],
            )
            pairs = list(move_map.keys())
            p = np.array([probs[idx, f, t].item() for f, t in pairs], dtype=np.float32)

            if add_noise and len(search_paths[original_i]) == 1 and len(pairs) > 1:
                p = 0.75 * p + 0.25 * np.random.dirichlet([0.3] * len(pairs))
                p = p / p.sum()

            for (f, t), pr in zip(pairs, p):
                node.children[move_map[(f, t)]] = MCTSNode(prior=pr)
            node.expanded, leaf_values[original_i] = True, v

    for board, moves_pushed in zip(boards, moves_pushed_list):
        for _ in range(moves_pushed):
            board.pop()

    for i, path in enumerate(search_paths):
        v = leaf_values[i]
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += v
            v = -v


def root_policy_from_visits(root, temperature=1.0):
    moves = list(root.children.keys())
    visits = torch.tensor(
        [child.visit_count for child in root.children.values()], dtype=torch.float32
    )

    if temperature == 0:
        probs = torch.zeros_like(visits)
        probs[torch.argmax(visits)] = 1.0
    else:
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()

    return moves, probs


def root_converged(root, min_visit_share=0.95, min_visits=8):
    if root.visit_count < min_visits or not root.children:
        return False
    return (
        max(c.visit_count for c in root.children.values()) / root.visit_count
        >= min_visit_share
    )


def play_games_batched(
    model, device, num_games=128, sims=40, max_moves=120, sample_moves=15
):
    model.eval()
    boards = [chess.Board() for _ in range(num_games)]
    roots = [MCTSNode(prior=1.0) for _ in range(num_games)]
    trajectories = [[] for _ in range(num_games)]
    finished = [False] * num_games

    for ply in range(max_moves):
        active_indices = [i for i, f in enumerate(finished) if not f]
        if not active_indices:
            break

        active_boards = [boards[i] for i in active_indices]
        active_roots = [roots[i] for i in active_indices]

        for root in active_roots:
            if root.expanded:
                add_root_noise(root)

        batched_mcts_sim(active_roots, active_boards, model, device, add_noise=True)

        sim_indices = list(range(len(active_roots)))
        for _ in range(sims - 1):
            sim_indices = [
                i for i in sim_indices if not root_converged(active_roots[i])
            ]
            if not sim_indices:
                break
            batched_mcts_sim(
                [active_roots[i] for i in sim_indices],
                [active_boards[i] for i in sim_indices],
                model,
                device,
                add_noise=False,
            )

        for idx, original_i in enumerate(active_indices):
            board, root = boards[original_i], roots[original_i]
            if not root.children or board.is_game_over(claim_draw=True):
                finished[original_i] = True
                continue

            moves, probs = root_policy_from_visits(
                root, temperature=1.0 if ply < sample_moves else 0.5
            )
            policy_pairs = np.array(
                [
                    (
                        canonical_square(move.from_square, board),
                        canonical_square(move.to_square, board),
                    )
                    for move in moves
                ],
                dtype=np.uint8,
            )
            policy_probs = probs.numpy().astype(np.float32)
            legal_pairs = np.array(
                list(legal_moves_by_square_pair(board).keys()), dtype=np.uint8
            )

            trajectories[original_i].append(
                {
                    "board_tokens": board_to_tokens(board),
                    "legal_pairs": legal_pairs,
                    "policy_pairs": policy_pairs,
                    "policy_probs": policy_probs,
                    "turn": board.turn,
                }
            )
            chosen = (
                random.choices(moves, weights=probs.tolist(), k=1)[0]
                if ply < sample_moves
                else moves[int(torch.argmax(probs).item())]
            )
            board.push(chosen)
            roots[original_i] = root.children[chosen]

    samples = []
    for i, board in enumerate(boards):
        out = board.outcome(claim_draw=True)
        outcome_white = (
            1.0
            if out and out.winner == chess.WHITE
            else -1.0 if out and out.winner == chess.BLACK else 0.0
        )
        for step in trajectories[i]:
            value = outcome_white if step["turn"] == chess.WHITE else -outcome_white
            samples.append(
                (
                    np.array(step["board_tokens"], dtype=np.uint8),
                    step["legal_pairs"],
                    step["policy_pairs"],
                    step["policy_probs"],
                    value,
                )
            )
    return samples
