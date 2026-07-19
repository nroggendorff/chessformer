import math

import torch

from encoding import BOARD_SQUARES, board_to_input, legal_moves_by_square_pair
from model import MAX_PIECES, piece_gather


def _piece_move_options(board):
    move_map = legal_moves_by_square_pair(board)
    by_from = {}
    for (frm, to), move in move_map.items():
        by_from.setdefault(frm, []).append(to)
    return by_from, move_map


def _select_destination(masked_logits, temperature):
    if temperature <= 0:
        return int(masked_logits.argmax().item())
    return int(
        torch.multinomial(torch.softmax(masked_logits / temperature, dim=-1), 1).item()
    )


def _piece_dest_masks(heatmap_b, piece_squares_b, piece_mask_b, by_from, device):
    for slot in range(MAX_PIECES):
        if not piece_mask_b[slot]:
            continue
        frm = int(piece_squares_b[slot])
        dests = by_from.get(frm)
        if not dests:
            continue
        dest_mask = torch.full((BOARD_SQUARES,), False, device=device)
        dest_mask[dests] = True
        yield slot, frm, dests, dest_mask, heatmap_b[slot].masked_fill(~dest_mask, -1e4)


def _best_per_piece_candidates(
    heatmap_b, piece_squares_b, piece_mask_b, by_from, move_map, device, temperature
):
    candidates = []
    for slot, frm, _, dest_mask, masked_logits in _piece_dest_masks(
        heatmap_b, piece_squares_b, piece_mask_b, by_from, device
    ):
        best_to = _select_destination(masked_logits, temperature)
        candidates.append(
            (slot, best_to, move_map[(frm, best_to)], dest_mask, masked_logits[best_to])
        )
    return candidates


def _top_fraction_candidates(
    heatmap_b, piece_squares_b, piece_mask_b, by_from, move_map, device, top_fraction
):
    candidates = [
        (slot, to, move_map[(frm, to)], dest_mask, masked_logits[to])
        for slot, frm, dests, dest_mask, masked_logits in _piece_dest_masks(
            heatmap_b, piece_squares_b, piece_mask_b, by_from, device
        )
        for to in dests
    ]
    keep = max(1, math.ceil(top_fraction * len(candidates)))
    return sorted(candidates, key=lambda c: c[4], reverse=True)[:keep]


@torch.inference_mode()
def batched_policy_step(
    boards,
    model,
    device,
    temperature=0.0,
    top_fraction=None,
    max_candidates=None,
):
    board_inputs = torch.tensor(
        [board_to_input(board) for board in boards], dtype=torch.long, device=device
    )
    heatmap, value = model(board_inputs)
    piece_squares, piece_mask = piece_gather(board_inputs[:, :BOARD_SQUARES])
    piece_squares, piece_mask = piece_squares.cpu(), piece_mask.cpu()

    per_board_candidates = [[] for _ in boards]
    child_inputs = []
    for b, board in enumerate(boards):
        by_from, move_map = _piece_move_options(board)
        candidates = (
            _top_fraction_candidates(
                heatmap[b],
                piece_squares[b],
                piece_mask[b],
                by_from,
                move_map,
                device,
                top_fraction,
            )
            if top_fraction is not None
            else _best_per_piece_candidates(
                heatmap[b],
                piece_squares[b],
                piece_mask[b],
                by_from,
                move_map,
                device,
                temperature,
            )
        )
        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = sorted(candidates, key=lambda c: c[4], reverse=True)[
                :max_candidates
            ]
        for slot, best_to, move, dest_mask, _ in candidates:
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
