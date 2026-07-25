import random

import chess
import numpy as np

from encoding import board_to_input, legal_moves_by_square_pair
from tree_search import choose_move, run_mcts, visit_policy_pairs


def game_over(board, ply, draw_check_interval=4):
    return board.outcome(claim_draw=ply % draw_check_interval == 0) is not None


def play_games_batched(
    model,
    device,
    num_games=128,
    max_moves=120,
    sample_moves=15,
    temperature=1.0,
    temperature_floor=0.1,
    decisive_weight=1.5,
    mcts_simulations=200,
    opponent_mcts_simulations=100,
    sims_per_wave=8,
    target_batch_size=None,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    root_noise_frac=0.25,
    opponent_model=None,
):
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()
    boards = [chess.Board() for _ in range(num_games)]
    learner_color = [
        chess.WHITE if random.random() < 0.5 else chess.BLACK for _ in range(num_games)
    ]
    trajectories: list[list[dict]] = [[] for _ in range(num_games)]
    finished = [False] * num_games

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

        if learner_idx:
            roots = run_mcts(
                [boards[i] for i in learner_idx],
                model,
                device,
                num_simulations=mcts_simulations,
                sims_per_wave=sims_per_wave,
                target_batch_size=target_batch_size,
                c_puct=c_puct,
                add_root_noise=True,
                root_dirichlet_alpha=dirichlet_alpha,
                root_noise_frac=root_noise_frac,
            )
            for original_i, root in zip(learner_idx, roots):
                board = boards[original_i]
                move = choose_move(root, temperature_now)
                if move is None:
                    finished[original_i] = True
                    continue
                policy_pairs = visit_policy_pairs(root)
                trajectories[original_i].append(
                    {
                        "board_input": board_to_input(
                            board, legal_moves=root.legal_moves
                        ),
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
                board.push(move)
                if game_over(board, ply):
                    finished[original_i] = True

        if opponent_idx:
            roots = run_mcts(
                [boards[i] for i in opponent_idx],
                opponent_model,
                device,
                num_simulations=opponent_mcts_simulations,
                sims_per_wave=sims_per_wave,
                target_batch_size=target_batch_size,
                c_puct=c_puct,
                add_root_noise=False,
            )
            for original_i, root in zip(opponent_idx, roots):
                board = boards[original_i]
                move = choose_move(root, temperature_floor)
                if move is None:
                    finished[original_i] = True
                    continue
                board.push(move)
                if game_over(board, ply):
                    finished[original_i] = True

    samples = []
    for board, trajectory, is_finished in zip(boards, trajectories, finished):
        if not trajectory:
            continue
        winner = board.outcome(claim_draw=True).winner if is_finished else None
        policy_weight = decisive_weight if winner is not None else 1.0
        value_weight = policy_weight if is_finished else 0.0
        for step in trajectory:
            value_target = (
                0.0
                if winner is None
                else float(1.0 if winner == step["turn"] else -1.0)
            )
            samples.append(
                (
                    np.array(step["board_input"], dtype=np.uint8),
                    step["legal_pairs"],
                    step["policy_pairs"],
                    step["policy_probs"],
                    value_target,
                    policy_weight,
                    value_weight,
                )
            )

    return samples
