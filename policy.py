import math

import chess
import numpy as np
import torch

from encoding import BOARD_SQUARES, board_to_input, legal_moves_by_square_pair
from model import MAX_PIECES, piece_gather

PROMOTION_PIECE_TYPES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
MASK_VALUE = -1e4


def _promotion_variants(move):
    return [
        chess.Move(move.from_square, move.to_square, promotion=piece_type)
        for piece_type in PROMOTION_PIECE_TYPES
    ]


def _pushed(board, move):
    child = board.copy()
    child.push(move)
    return child


def resolve_promotions(boards, moves, model, device):
    promo_idx = [b for b, move in enumerate(moves) if move.promotion is not None]
    if not promo_idx:
        return moves

    variants_by_idx = {b: _promotion_variants(moves[b]) for b in promo_idx}
    variant_inputs = [
        board_to_input(_pushed(boards[b], variant))
        for b in promo_idx
        for variant in variants_by_idx[b]
    ]
    _, variant_values = model(
        torch.tensor(variant_inputs, dtype=torch.long, device=device), value_only=True
    )
    variant_values = variant_values.cpu()

    offset = 0
    for b in promo_idx:
        variants = variants_by_idx[b]
        best = int((-variant_values[offset : offset + len(variants)]).argmax().item())
        moves[b] = variants[best]
        offset += len(variants)

    return moves


def _piece_move_options(board):
    move_map = legal_moves_by_square_pair(board)
    by_from = {}
    for (frm, to), move in move_map.items():
        by_from.setdefault(frm, []).append(to)
    return by_from, move_map


def _dest_mask_array(piece_squares, piece_mask, move_options):
    mask = np.zeros((len(move_options), MAX_PIECES, BOARD_SQUARES), dtype=bool)
    for b, (by_from, _) in enumerate(move_options):
        for slot in range(MAX_PIECES):
            if not piece_mask[b, slot]:
                continue
            dests = by_from.get(int(piece_squares[b, slot]))
            if dests:
                mask[b, slot, dests] = True
    return mask


def _select_best_per_piece(masked_logits, temperature):
    B, S, D = masked_logits.shape
    if temperature <= 0:
        return masked_logits.argmax(dim=-1)
    probs = torch.softmax(masked_logits.reshape(B * S, D) / temperature, dim=-1)
    return torch.multinomial(probs, 1).view(B, S)


def _per_piece_candidates(
    masked_logits, dest_mask, piece_squares, move_options, temperature
):
    has_dest = dest_mask.any(dim=-1).cpu().numpy()
    best_to = _select_best_per_piece(masked_logits, temperature)
    selected_logits = (
        torch.gather(masked_logits, -1, best_to.unsqueeze(-1))
        .squeeze(-1)
        .float()
        .cpu()
        .numpy()
    )
    best_to = best_to.cpu().numpy()

    return [
        [
            (
                slot,
                int(best_to[b, slot]),
                move_options[b][1][
                    (int(piece_squares[b, slot]), int(best_to[b, slot]))
                ],
                float(selected_logits[b, slot]),
            )
            for slot in range(masked_logits.size(1))
            if has_dest[b, slot]
        ]
        for b in range(masked_logits.size(0))
    ]


def _top_fraction_candidates(
    masked_logits, dest_mask, piece_squares, move_options, top_fraction
):
    B, S, D = masked_logits.shape
    num_valid = dest_mask.reshape(B, S * D).sum(dim=-1).cpu().numpy()
    sorted_vals, sorted_idx = torch.sort(
        masked_logits.reshape(B, S * D), dim=-1, descending=True
    )
    sorted_vals, sorted_idx = (
        sorted_vals.float().cpu().numpy(),
        sorted_idx.cpu().numpy(),
    )

    candidates = []
    for b in range(B):
        keep = max(1, math.ceil(top_fraction * num_valid[b]))
        board_candidates = []
        for rank in range(keep):
            slot, to = divmod(int(sorted_idx[b, rank]), D)
            frm = int(piece_squares[b, slot])
            board_candidates.append(
                (slot, to, move_options[b][1][(frm, to)], float(sorted_vals[b, rank]))
            )
        candidates.append(board_candidates)
    return candidates


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
    piece_squares, piece_mask = piece_squares.cpu().numpy(), piece_mask.cpu().numpy()

    move_options = [_piece_move_options(board) for board in boards]
    dest_mask = torch.from_numpy(
        _dest_mask_array(piece_squares, piece_mask, move_options)
    ).to(device)
    masked_logits = heatmap.masked_fill(~dest_mask, MASK_VALUE)

    per_board_candidates_raw = (
        _top_fraction_candidates(
            masked_logits, dest_mask, piece_squares, move_options, top_fraction
        )
        if top_fraction is not None
        else _per_piece_candidates(
            masked_logits, dest_mask, piece_squares, move_options, temperature
        )
    )

    per_board_candidates, child_inputs = [], []
    for b, board in enumerate(boards):
        candidates = per_board_candidates_raw[b]
        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = sorted(candidates, key=lambda c: c[3], reverse=True)[
                :max_candidates
            ]
        board_candidates = []
        for slot, to, move, _ in candidates:
            child = board.copy()
            child.push(move)
            board_candidates.append((slot, to, move))
            child_inputs.append(board_to_input(child))
        per_board_candidates.append(board_candidates)

    if not child_inputs:
        raise ValueError("batched_policy_step called with no legal moves available")

    _, child_values = model(
        torch.tensor(child_inputs, dtype=torch.long, device=device), value_only=True
    )
    child_values = child_values.cpu()

    moves, chosen_slots, chosen_tos = [], [], []
    offset = 0
    for candidates in per_board_candidates:
        desirability = -child_values[offset : offset + len(candidates)]
        offset += len(candidates)
        choice = (
            int(desirability.argmax().item())
            if temperature <= 0
            else int(
                torch.multinomial(
                    torch.softmax(desirability / temperature, dim=-1), 1
                ).item()
            )
        )
        slot, to, move = candidates[choice]
        moves.append(move)
        chosen_slots.append(slot)
        chosen_tos.append(to)

    batch_idx = torch.arange(len(boards), device=device)
    chosen_logits = masked_logits[batch_idx, torch.tensor(chosen_slots, device=device)]
    tempered_logits = chosen_logits / temperature if temperature > 0 else chosen_logits
    log_probs = (
        torch.log_softmax(tempered_logits, dim=-1)[
            batch_idx, torch.tensor(chosen_tos, device=device)
        ]
        .cpu()
        .tolist()
    )
    values = value.cpu().tolist()

    moves = resolve_promotions(boards, moves, model, device)

    return moves, values, torch.from_numpy(piece_mask), log_probs
