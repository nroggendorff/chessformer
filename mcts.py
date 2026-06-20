import math
import random

import chess
import numpy as np
import torch

from encoding import MAX_LEGAL_MOVES, board_to_tokens, move_to_index


class MCTSNode:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "expanded",
        "legal_moves",
        "action_ids",
        "terminal_value",
    )

    def __init__(self, prior=1.0):
        self.prior = float(prior)
        self.visit_count, self.value_sum = 0, 0.0
        self.children = {}
        self.expanded = False
        self.legal_moves = self.action_ids = None
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

    eval_indices, eval_tokens, eval_legal_moves = [], [], []
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
            if node.legal_moves is None:
                node.legal_moves = list(board.legal_moves)
            eval_legal_moves.append(node.legal_moves)

    if eval_indices:
        num_eval = len(eval_indices)
        pad_size = 1 if num_eval == 0 else 2 ** (num_eval - 1).bit_length()

        b_tokens = torch.tensor(eval_tokens, dtype=torch.long, device=device)

        if num_eval < pad_size:
            pad_tokens = torch.zeros(
                (pad_size - num_eval, b_tokens.shape[1]),
                dtype=torch.long,
                device=device,
            )
            b_tokens = torch.cat([b_tokens, pad_tokens], dim=0)

        b_actions_cpu = torch.zeros((pad_size, MAX_LEGAL_MOVES), dtype=torch.long)
        mask_cpu = torch.zeros((pad_size, MAX_LEGAL_MOVES), dtype=torch.bool)

        for idx, original_i in enumerate(eval_indices):
            node = search_paths[original_i][-1]
            if node.action_ids is None:
                node.action_ids = torch.tensor(
                    [move_to_index(m) for m in eval_legal_moves[idx]], dtype=torch.long
                )
            n = len(node.action_ids)
            b_actions_cpu[idx, :n] = node.action_ids
            mask_cpu[idx, :n] = True

        b_actions = b_actions_cpu.to(device, non_blocking=True)
        mask_device = mask_cpu.to(device, non_blocking=True)

        logits, val_preds = model(b_tokens, b_actions)

        logits = logits[:num_eval]
        val_preds = val_preds[:num_eval]
        mask_device = mask_device[:num_eval]

        logits = logits.masked_fill(~mask_device, -1e4)
        probs, val_preds = torch.softmax(logits, dim=-1).cpu(), val_preds.cpu().tolist()

        for idx, original_i in enumerate(eval_indices):
            node, v, p, mvs = (
                search_paths[original_i][-1],
                val_preds[idx],
                probs[idx],
                eval_legal_moves[idx],
            )
            p = (
                (
                    0.75 * p[: len(mvs)].numpy()
                    + 0.25 * np.random.dirichlet([0.3] * len(mvs))
                )
                if (add_noise and len(search_paths[original_i]) == 1 and len(mvs) > 1)
                else p[: len(mvs)].numpy()
            )

            for mv_idx, mv in enumerate(mvs):
                node.children[mv] = MCTSNode(prior=p[mv_idx])
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
    legal_moves = list(root.children.keys())
    visits = torch.tensor(
        [child.visit_count for child in root.children.values()], dtype=torch.float32
    )

    if temperature == 0:
        probs = torch.zeros_like(visits)
        probs[torch.argmax(visits)] = 1.0
    else:
        visits = visits ** (1.0 / temperature)
        probs = visits / visits.sum()

    if root.action_ids is None:
        root.action_ids = torch.tensor(
            [move_to_index(m) for m in legal_moves], dtype=torch.long
        )
    return legal_moves, root.action_ids, probs


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
    trajectories = [[] for _ in range(num_games)]
    finished = [False] * num_games

    for ply in range(max_moves):
        active_indices = [i for i, f in enumerate(finished) if not f]
        if not active_indices:
            break

        active_boards = [boards[i] for i in active_indices]
        roots = [MCTSNode(prior=1.0) for _ in active_boards]
        batched_mcts_sim(roots, active_boards, model, device, add_noise=True)

        for _ in range(sims - 1):
            if all(root_converged(r) for r in roots):
                break
            batched_mcts_sim(roots, active_boards, model, device, add_noise=False)

        for idx, original_i in enumerate(active_indices):
            board, root = boards[original_i], roots[idx]
            if not root.children or board.is_game_over(claim_draw=True):
                finished[original_i] = True
                continue

            legal_moves, action_ids, probs = root_policy_from_visits(
                root, temperature=1.0 if ply < sample_moves else 0.5
            )
            trajectories[original_i].append(
                {
                    "board_tokens": board_to_tokens(board),
                    "action_ids": action_ids.numpy().astype(np.int16),
                    "probs": probs.numpy().astype(np.float32),
                    "turn": board.turn,
                }
            )
            board.push(
                random.choices(legal_moves, weights=probs.tolist(), k=1)[0]
                if ply < sample_moves
                else legal_moves[int(torch.argmax(probs).item())]
            )

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
                    step["action_ids"],
                    step["probs"],
                    value,
                )
            )
    return samples
