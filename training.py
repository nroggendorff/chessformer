import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import amp_dtype
from encoding import BOARD_SQUARES
from model import MAX_PIECES, piece_gather


def _piece_targets_and_mask(piece_squares, samples):
    target = np.zeros((len(samples), MAX_PIECES, BOARD_SQUARES), dtype=np.float32)
    legal = np.zeros((len(samples), MAX_PIECES, BOARD_SQUARES), dtype=bool)
    for b, (_, legal_pairs, policy_pairs, policy_probs, *_) in enumerate(samples):
        slot_of_square = {int(sq): slot for slot, sq in enumerate(piece_squares[b])}
        for frm, to in legal_pairs:
            slot = slot_of_square.get(int(frm))
            if slot is not None:
                legal[b, slot, to] = True
        for (frm, to), p in zip(policy_pairs, policy_probs):
            slot = slot_of_square.get(int(frm))
            if slot is not None:
                target[b, slot, to] += p
    mass = target.sum(axis=-1, keepdims=True)
    per_piece_target = np.divide(
        target, mass, out=np.zeros_like(target), where=mass > 0
    )
    return per_piece_target, target, legal, mass[..., 0]


def train_batch(model, opt, scaler, samples, device, entropy_coef=0.01):
    model.train()

    boards = torch.from_numpy(np.stack([s[0] for s in samples])).long().to(device)
    piece_squares, _ = piece_gather(boards[:, :BOARD_SQUARES])
    target_policy_np, raw_target_np, legal_mask_np, slot_mass_np = (
        _piece_targets_and_mask(piece_squares.cpu().numpy(), samples)
    )
    target_policy = torch.from_numpy(target_policy_np).to(device)
    raw_target = torch.from_numpy(raw_target_np).to(device)
    legal_mask = torch.from_numpy(legal_mask_np).to(device)
    slot_w = torch.from_numpy(slot_mass_np).to(device)
    slot_w_denom = slot_w.sum(dim=-1).clamp(min=1e-6)
    active = slot_w > 0
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
        heatmaps, values = model(boards)
        masked = heatmaps.masked_fill(~legal_mask, -1e4)
        log_probs = F.log_softmax(masked, dim=-1).clamp(min=-20.0)
        per_piece_log_prob = (target_policy * log_probs).sum(dim=-1)
        per_piece_ce = -(per_piece_log_prob * slot_w).sum(dim=-1) / slot_w_denom

        global_log_probs = F.log_softmax(masked.flatten(1), dim=-1).clamp(min=-20.0)
        global_ce = -(raw_target.flatten(1) * global_log_probs).sum(dim=-1)

        policy_loss = ((per_piece_ce + global_ce) * policy_weights).mean()

        piece_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy = ((piece_entropy * active).sum(dim=-1) / active_count).mean()

        value_loss = (
            value_weights * F.mse_loss(values, target_values, reduction="none")
        ).mean()
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            raise RuntimeError(
                f"Non-finite weights, aborting before they get saved: {bad[:5]}"
            )
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    with torch.no_grad():
        has_policy = policy_weights > 0
        target_entropy = -(
            target_policy * torch.log(target_policy.clamp(min=1e-12))
        ).sum(dim=-1)
        kl_per_sample = ((-per_piece_log_prob.detach() - target_entropy) * active).sum(
            dim=-1
        ) / active_count
        top1_match = (
            target_policy.argmax(dim=-1) == log_probs.detach().argmax(dim=-1)
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
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(total_norm):
            opt.step()
        else:
            print(
                f"skipping optimizer step: non-finite grad norm ({total_norm.item()})"
            )
        opt.zero_grad(set_to_none=True)

    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        kl_div.item(),
        top1_acc.item(),
    )
