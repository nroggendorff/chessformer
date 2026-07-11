import numpy as np
import torch
import torch.nn.functional as F

from encoding import board_to_input, legal_moves_by_square_pair


def legal_mask_and_move_maps(boards):
    move_maps = [legal_moves_by_square_pair(board) for board in boards]
    mask = torch.zeros(len(boards), 64, 64, dtype=torch.bool)
    for i, move_map in enumerate(move_maps):
        pairs = np.array(list(move_map.keys()))
        mask[i, pairs[:, 0], pairs[:, 1]] = True
    return move_maps, mask


def joint_move_log_probs(heatmaps, legal_mask):
    masked = heatmaps.masked_fill(~legal_mask, -1e4)
    log_p_from = F.log_softmax(torch.logsumexp(masked, dim=-1), dim=-1)
    log_p_to_given_from = F.log_softmax(masked, dim=-1)
    return log_p_from[:, :, None] + log_p_to_given_from


@torch.inference_mode()
def batched_policy_step(boards, model, device, temperature=0.0):
    move_maps, legal_mask = legal_mask_and_move_maps(boards)
    heatmaps, values = model(
        torch.tensor(
            [board_to_input(board) for board in boards], dtype=torch.long, device=device
        )
    )
    legal_mask = legal_mask.to(device)
    flat_log_probs = joint_move_log_probs(heatmaps, legal_mask).view(len(boards), -1)

    choices = (
        flat_log_probs.argmax(dim=-1)
        if temperature == 0
        else torch.multinomial(
            torch.softmax(flat_log_probs / temperature, dim=-1), 1
        ).squeeze(-1)
    )
    from_sq, to_sq = (choices // 64).tolist(), (choices % 64).tolist()
    return (
        [move_maps[i][(f, t)] for i, (f, t) in enumerate(zip(from_sq, to_sq))],
        values.cpu().tolist(),
        legal_mask,
        flat_log_probs.gather(-1, choices[:, None]).squeeze(-1).cpu().tolist(),
    )
