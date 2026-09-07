import math

import chess
import chess.engine
import numpy as np
import torch

from data_generation import PIECE_VALUES
from encoding import board_to_input, legal_moves_by_square_pair
from evaluation import anchor_move
from tree_search import (
    MCTSNode,
    choose_move,
    forced_win_move,
    game_over,
    game_result,
    improved_policy_pairs,
    run_mcts,
)


@torch.inference_mode()
def _bootstrap_timeout_values(boards, indices, model, device):
    if not indices:
        return {}
    board_inputs = torch.tensor(
        [board_to_input(boards[i]) for i in indices],
        dtype=torch.long,
        device=device,
    )
    _, values, _ = model(board_inputs, value_only=True)
    return dict(zip(indices, values.float().cpu().tolist()))


def material_balance(board, color):
    return sum(
        value
        * (
            len(board.pieces(piece_type, color))
            - len(board.pieces(piece_type, not color))
        )
        for piece_type, value in PIECE_VALUES.items()
    )


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
    fpu_reduction=0.25,
    dirichlet_alpha=0.3,
    root_noise_frac=0.25,
    opponent_model=None,
    stockfish_engine=None,
    stockfish_anchor=None,
    stockfish_limit=None,
    stockfish_movetime=0.1,
    material_scale=0.0,
    material_value_weight=0.5,
    resign_threshold=None,
    resign_streak=2,
    add_root_noise=True,
    value_smoothing=0.0,
    record_trajectory=True,
    include_policy_q_threshold=0.85,
    opening_moves_per_game=None,
    target_beta=0.0,
):
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()
    self_play_mode = opponent_model is None and stockfish_anchor is None

    boards = [chess.Board() for _ in range(num_games)]
    if opening_moves_per_game is not None:
        for board, moves in zip(boards, opening_moves_per_game):
            for uci in moves:
                board.push_uci(uci)
    roots = [MCTSNode(board.copy()) for board in boards]
    learner_color = [
        chess.WHITE if i % 2 == 0 else chess.BLACK for i in range(num_games)
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
            i for i in active if self_play_mode or boards[i].turn == learner_color[i]
        ]
        opponent_idx = [i for i in active if i not in learner_idx]
        temperature_now = temperature if ply < sample_moves else temperature_floor

        if learner_idx:
            run_mcts(
                [roots[i] for i in learner_idx],
                model,
                device,
                num_simulations=mcts_simulations,
                sims_per_wave=sims_per_wave,
                target_batch_size=target_batch_size,
                max_batch_size=max_batch_size,
                c_puct=c_puct,
                fpu_reduction=fpu_reduction,
                add_root_noise=add_root_noise,
                root_dirichlet_alpha=dirichlet_alpha,
                root_noise_frac=root_noise_frac,
            )
            for i in learner_idx:
                board, root = boards[i], roots[i]
                move = choose_move(root, temperature_now)
                if move is None:
                    finished[i] = True
                    continue

                root_q = (
                    -root.value_sum / root.visit_count if root.visit_count > 0 else 0.0
                )

                if record_trajectory:
                    proven_win = forced_win_move(root) is not None
                    policy_pairs = improved_policy_pairs(root, board.turn, target_beta)
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
                            "include_policy": proven_win
                            or abs(root_q) < include_policy_q_threshold,
                        }
                    )

                if resign_threshold is not None and root.visit_count > 0:
                    q = root_q
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
                    if self_play_mode
                    else MCTSNode(board.copy())
                )
                if game_over(board):
                    finished[i] = True

        if opponent_idx and stockfish_anchor is not None:
            for i in opponent_idx:
                board = boards[i]
                move = anchor_move(
                    stockfish_engine,
                    stockfish_anchor,
                    stockfish_limit,
                    board,
                    stockfish_movetime,
                )
                if move is None:
                    finished[i] = True
                    continue
                board.push(move)
                roots[i] = MCTSNode(board.copy())
                if game_over(board):
                    finished[i] = True
        elif opponent_idx:
            run_mcts(
                [roots[i] for i in opponent_idx],
                opponent_model,
                device,
                num_simulations=opponent_mcts_simulations,
                sims_per_wave=sims_per_wave,
                target_batch_size=target_batch_size,
                max_batch_size=max_batch_size,
                c_puct=c_puct,
                fpu_reduction=fpu_reduction,
                add_root_noise=False,
                root_dirichlet_alpha=dirichlet_alpha,
                root_noise_frac=root_noise_frac,
            )
            for i in opponent_idx:
                board, root = boards[i], roots[i]
                move = choose_move(root, temperature_now)
                if move is None:
                    finished[i] = True
                    continue
                board.push(move)
                roots[i] = MCTSNode(board.copy())
                if game_over(board):
                    finished[i] = True

    resolved_flags, winners = [], [None] * num_games
    for i in range(num_games):
        finished_game, winner = game_result(boards[i])
        winners[i] = winner if finished_game else adjudicated_winner[i]
        resolved_flags.append(finished_game or adjudicated_winner[i] is not None)

    timeout_idx = [
        i for i in range(num_games) if not resolved_flags[i] and trajectories[i]
    ]
    timeout_values = (
        {}
        if material_scale > 0
        else _bootstrap_timeout_values(boards, timeout_idx, model, device)
    )
    timeout_material = (
        {
            i: math.tanh(material_balance(boards[i], chess.WHITE) / material_scale)
            for i in timeout_idx
        }
        if material_scale > 0
        else {}
    )

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
        adjudicated = adjudicated_winner[i] is not None
        decisive_game = winner is not None and not adjudicated
        value_weight = (
            (decisive_weight if decisive_game else 1.0)
            if resolved
            else (material_value_weight if material_scale > 0 else timeout_value_weight)
        )
        bootstrap = timeout_values.get(i)
        white_material = timeout_material.get(i)
        for step in trajectory:
            if resolved:
                value_target = (
                    0.0
                    if winner is None
                    else float(
                        (1.0 - value_smoothing)
                        if winner == step["turn"]
                        else -(1.0 - value_smoothing)
                    )
                )
            elif white_material is not None:
                value_target = (
                    white_material if step["turn"] == chess.WHITE else -white_material
                )
            else:
                value_target = bootstrap if step["turn"] == board.turn else -bootstrap
            policy_weight = 0.0
            if step["include_policy"]:
                policy_weight = (
                    decisive_weight if decisive_game and step["turn"] == winner else 1.0
                )
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

    unresolved_positions = [
        (boards[i].fen(), learner_color[i] == chess.WHITE)
        for i in range(num_games)
        if not resolved_flags[i]
    ]

    stats = {
        "unresolved_positions": unresolved_positions,
        "games": num_games,
        "decisive": decisive,
        "drawn": drawn,
        "unresolved": num_games - decisive - drawn,
        "learner_wins": sum(
            resolved_flags[i] and winners[i] == learner_color[i]
            for i in range(num_games)
        ),
        "opponent_wins": sum(
            resolved_flags[i]
            and winners[i] is not None
            and winners[i] != learner_color[i]
            for i in range(num_games)
        ),
    }
    return samples, stats
