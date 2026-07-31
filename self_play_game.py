import random

import chess
import numpy as np
import torch

from encoding import board_to_input, legal_moves_by_square_pair
from tree_search import MCTSNode, choose_move, run_mcts, visit_policy_pairs


@torch.inference_mode()
def _bootstrap_timeout_values(boards, indices, model, device):
    if not indices:
        return {}
    board_inputs = torch.tensor(
        [board_to_input(boards[i]) for i in indices],
        dtype=torch.long,
        device=device,
    )
    _, values = model(board_inputs, value_only=True)
    return dict(zip(indices, values.float().cpu().tolist()))


def _advance_root(root, move, board):
    child = root.children.get(move)
    if child is None:
        return MCTSNode(board.copy())
    child.ensure_board()
    child.parent = None
    return child


def play_games_batched(
    model,
    device,
    num_games=128,
    max_moves=120,
    sample_moves=15,
    temperature=1.0,
    temperature_floor=0.1,
    decisive_weight=1.5,
    timeout_value_weight=0.5,
    mcts_simulations=200,
    opponent_mcts_simulations=100,
    sims_per_wave=8,
    target_batch_size=None,
    max_batch_size=None,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    root_noise_frac=0.25,
    opponent_model=None,
    resign_threshold=None,
    resign_streak=2,
):
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()

    boards = [chess.Board() for _ in range(num_games)]
    roots = [MCTSNode(board.copy()) for board in boards]
    learner_color = [
        chess.WHITE if random.random() < 0.5 else chess.BLACK for _ in range(num_games)
    ]
    trajectories: list[list[dict]] = [[] for _ in range(num_games)]
    finished = [False] * num_games
    adjudicated_winner = [None] * num_games
    losing_streak = [[0, 0] for _ in range(num_games)]

    for ply in range(max_moves):
        active = [i for i, f in enumerate(finished) if not f]
        if not active:
            break

        learner_idx = [
            i
            for i in active
            if opponent_model is None or boards[i].turn == learner_color[i]
        ]
        opponent_idx = [i for i in active if i not in learner_idx]
        temperature_now = temperature if ply < sample_moves else temperature_floor

        for indices, search_model, sims, noise, record, temp in (
            (learner_idx, model, mcts_simulations, True, True, temperature_now),
            (
                opponent_idx,
                opponent_model,
                opponent_mcts_simulations,
                False,
                False,
                temperature_floor,
            ),
        ):
            if not indices:
                continue
            run_mcts(
                [roots[i] for i in indices],
                search_model,
                device,
                num_simulations=sims,
                sims_per_wave=sims_per_wave,
                target_batch_size=target_batch_size,
                max_batch_size=max_batch_size,
                c_puct=c_puct,
                add_root_noise=noise,
                root_dirichlet_alpha=dirichlet_alpha,
                root_noise_frac=root_noise_frac,
            )
            for i in indices:
                board, root = boards[i], roots[i]
                move = choose_move(root, temp)
                if move is None:
                    finished[i] = True
                    continue

                if record:
                    policy_pairs = visit_policy_pairs(root)
                    trajectories[i].append(
                        {
                            "board_input": board_to_input(board),
                            "legal_pairs": np.array(
                                list(
                                    legal_moves_by_square_pair(
                                        board, legal_moves=root.legal_moves
                                    ).keys()
                                ),
                                dtype=np.uint8,
                            ),
                            "policy_pairs": np.array(
                                list(policy_pairs.keys()), dtype=np.uint8
                            ).reshape(-1, 2),
                            "policy_probs": np.array(
                                list(policy_pairs.values()), dtype=np.float32
                            ),
                            "turn": board.turn,
                        }
                    )

                if resign_threshold is not None and root.visit_count > 0:
                    q = -root.value_sum / root.visit_count
                    streaks = losing_streak[i]
                    streaks[board.turn] = (
                        streaks[board.turn] + 1 if q < -resign_threshold else 0
                    )
                    if streaks[board.turn] >= resign_streak:
                        adjudicated_winner[i] = not board.turn
                        finished[i] = True
                        continue

                board.push(move)
                roots[i] = (
                    _advance_root(root, move, board)
                    if opponent_model is None
                    else MCTSNode(board.copy())
                )
                if board.outcome(claim_draw=True) is not None:
                    finished[i] = True

    resolved_flags, winners = [], [None] * num_games
    for i in range(num_games):
        outcome = boards[i].outcome(claim_draw=True)
        winners[i] = outcome.winner if outcome is not None else adjudicated_winner[i]
        resolved_flags.append(outcome is not None or adjudicated_winner[i] is not None)

    timeout_idx = [
        i for i in range(num_games) if not resolved_flags[i] and trajectories[i]
    ]
    timeout_values = _bootstrap_timeout_values(boards, timeout_idx, model, device)

    samples, decisive, drawn = [], 0, 0
    for i in range(num_games):
        board, trajectory, winner, resolved = (
            boards[i],
            trajectories[i],
            winners[i],
            resolved_flags[i],
        )
        if resolved:
            drawn += winner is None
            decisive += winner is not None
        if not trajectory:
            continue
        policy_weight = decisive_weight if winner is not None else 1.0
        value_weight = policy_weight if resolved else timeout_value_weight
        bootstrap = timeout_values.get(i)
        for step in trajectory:
            if resolved:
                value_target = (
                    0.0
                    if winner is None
                    else float(1.0 if winner == step["turn"] else -1.0)
                )
            else:
                value_target = bootstrap if step["turn"] == board.turn else -bootstrap
            samples.append(
                (
                    np.array(step["board_input"], dtype=np.int64),
                    step["legal_pairs"],
                    step["policy_pairs"],
                    step["policy_probs"],
                    value_target,
                    policy_weight,
                    value_weight,
                )
            )

    stats = {
        "games": num_games,
        "decisive": decisive,
        "drawn": drawn,
        "unresolved": num_games - decisive - drawn,
    }
    return samples, stats
