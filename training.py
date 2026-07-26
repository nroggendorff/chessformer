import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import amp_dtype
from encoding import BOARD_SQUARES, NUM_PIECE_TOKENS


def _board_targets(board_tokens, samples):
    recolored = np.where(
        board_tokens == 0,
        0,
        np.where(board_tokens <= 6, board_tokens + 6, board_tokens - 6),
    )
    target = np.zeros((len(samples), BOARD_SQUARES, NUM_PIECE_TOKENS), dtype=np.float32)
    active = np.zeros((len(samples), BOARD_SQUARES), dtype=bool)
    target[
        np.arange(len(samples))[:, None], np.arange(BOARD_SQUARES)[None, :], recolored
    ] = 1.0
    for b, (_, target_squares, target_tokens, target_weights, *_) in enumerate(samples):
        if len(target_squares):
            target[b, target_squares] = 0.0
            np.add.at(target[b], (target_squares, target_tokens), target_weights)
            active[b, np.unique(target_squares)] = True
    return target, active


def train_batch(model, opt, scaler, samples, device):
    model.train()

    boards = torch.from_numpy(np.stack([s[0] for s in samples])).long().to(device)
    target_np, active_np = _board_targets(
        boards[:, :BOARD_SQUARES].cpu().numpy(), samples
    )
    target = torch.from_numpy(target_np).to(device)
    active = torch.from_numpy(active_np).to(device)
    active_count = active.sum(dim=-1).clamp(min=1)
    target_values = torch.tensor(
        [s[4] for s in samples], dtype=torch.float32, device=device
    )
    policy_weights = torch.tensor(
        [s[5] for s in samples], dtype=torch.float32, device=device
    )
    value_weights = torch.tensor(
        [s[6] for s in samples], dtype=torch.float32, device=device
    )

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
        board_logits, values = model(boards)
        log_probs = F.log_softmax(board_logits, dim=-1).clamp(min=-20.0)
        per_square_log_prob = (target * log_probs).sum(dim=-1)
        sample_log_probs = (per_square_log_prob * active).sum(dim=-1) / active_count
        policy_loss = (-sample_log_probs * policy_weights).mean()

        square_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy = ((square_entropy * active).sum(dim=-1) / active_count).mean()

        value_loss = (
            value_weights * F.mse_loss(values, target_values, reduction="none")
        ).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    with torch.no_grad():
        has_policy = policy_weights > 0
        target_entropy = -(target * torch.log(target.clamp(min=1e-12))).sum(dim=-1)
        kl_per_sample = ((-per_square_log_prob.detach() - target_entropy) * active).sum(
            dim=-1
        ) / active_count
        top1_match = (
            target.argmax(dim=-1) == log_probs.detach().argmax(dim=-1)
        ).float()
        top1_per_sample = (top1_match * active).sum(dim=-1) / active_count
        if has_policy.any():
            kl_div = kl_per_sample[has_policy].mean()
            top1_acc = top1_per_sample[has_policy].mean()
        else:
            kl_div = kl_per_sample.mean() * 0.0
            top1_acc = top1_per_sample.mean() * 0.0

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        kl_div.item(),
        top1_acc.item(),
    )
