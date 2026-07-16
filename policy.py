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
def batched_policy_step(
    boards, model, device, temperature=0.0, use_diffuser=False, diffuser_steps=8
):
    move_maps, legal_mask = legal_mask_and_move_maps(boards)
    heatmaps, values = model(
        torch.tensor(
            [board_to_input(board) for board in boards], dtype=torch.long, device=device
        ),
        use_diffuser=use_diffuser,
        diffuser_steps=diffuser_steps,
    )
    legal_mask = legal_mask.to(device)
    flat_log_probs = joint_move_log_probs(heatmaps, legal_mask).view(len(boards), -1)
    flat_legal_mask = legal_mask.view(len(boards), -1)

    if temperature == 0:
        masked_log_probs = flat_log_probs.masked_fill(~flat_legal_mask, -float("inf"))
        choices = masked_log_probs.argmax(dim=-1)
        sampled_log_probs = flat_log_probs
    else:
        scaled_log_probs = flat_log_probs / temperature
        masked_scaled = scaled_log_probs.masked_fill(~flat_legal_mask, -float("inf"))
        sampled_log_probs = F.log_softmax(masked_scaled, dim=-1)
        choices = torch.multinomial(sampled_log_probs.exp(), 1).squeeze(-1)

    from_sq, to_sq = (choices // 64).tolist(), (choices % 64).tolist()
    return (
        [move_maps[i][(f, t)] for i, (f, t) in enumerate(zip(from_sq, to_sq))],
        values.cpu().tolist(),
        legal_mask,
        sampled_log_probs.gather(-1, choices[:, None]).squeeze(-1).cpu().tolist(),
    )
