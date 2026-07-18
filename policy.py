import torch

from encoding import BOARD_SQUARES, board_to_input, legal_moves_by_square_pair
from model import MAX_PIECES, piece_gather


def _piece_move_options(board):
    move_map = legal_moves_by_square_pair(board)
    by_from = {}
    for (frm, to), move in move_map.items():
        by_from.setdefault(frm, []).append(to)
    return by_from, move_map


@torch.inference_mode()
def batched_policy_step(
    boards, model, device, temperature=0.0, use_diffuser=None, diffuser_steps=None
):
    board_inputs = torch.tensor(
        [board_to_input(board) for board in boards], dtype=torch.long, device=device
    )
    heatmap, value = model(
        board_inputs, use_diffuser=use_diffuser, diffuser_steps=diffuser_steps
    )
    piece_squares, piece_mask = piece_gather(board_inputs[:, :BOARD_SQUARES])
    piece_squares, piece_mask = piece_squares.cpu(), piece_mask.cpu()

    per_board_candidates = [[] for _ in boards]
    child_inputs = []
    for b, board in enumerate(boards):
        by_from, move_map = _piece_move_options(board)
        for slot in range(MAX_PIECES):
            if not piece_mask[b, slot]:
                continue
            frm = int(piece_squares[b, slot])
            dests = by_from.get(frm)
            if not dests:
                continue
            dest_mask = torch.full((BOARD_SQUARES,), False, device=device)
            dest_mask[dests] = True
            best_to = int(
                heatmap[b, slot].masked_fill(~dest_mask, -1e4).argmax().item()
            )
            move = move_map[(frm, best_to)]
            child = board.copy()
            child.push(move)
            per_board_candidates[b].append((slot, best_to, move, dest_mask))
            child_inputs.append(board_to_input(child))

    if not child_inputs:
        raise ValueError("batched_policy_step called with no legal moves available")

    _, child_values = model(
        torch.tensor(child_inputs, dtype=torch.long, device=device), value_only=True
    )
    child_values = child_values.cpu()

    moves, values, log_probs = [], [], []
    offset = 0
    for b, candidates in enumerate(per_board_candidates):
        desirability = -child_values[offset : offset + len(candidates)]
        offset += len(candidates)
        if temperature <= 0:
            choice = int(desirability.argmax().item())
        else:
            choice = int(
                torch.multinomial(
                    torch.softmax(desirability / temperature, dim=-1), 1
                ).item()
            )
        slot, best_to, move, dest_mask = candidates[choice]
        moves.append(move)
        values.append(value[b].item())
        log_probs.append(
            torch.log_softmax(heatmap[b, slot].masked_fill(~dest_mask, -1e4), dim=-1)[
                best_to
            ].item()
        )

    return moves, values, piece_mask, log_probs
