import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import amp_dtype
from diffuser import board_images, vae_loss
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
    target = np.divide(target, mass, out=np.zeros_like(target), where=mass > 0)
    return target, legal, mass[..., 0] > 0


def train_batch(
    model,
    opt,
    scaler,
    samples,
    device,
    ref_model=None,
    kl_coef=0.0,
    clip_epsilon=0.2,
    vae=None,
    vae_opt=None,
    vae_kl_weight=1e-6,
    vae_loss_weight=0.1,
    use_diffuser=False,
    diffuser_steps=8,
):
    model.train()
    if vae is not None:
        vae.train()

    boards = torch.from_numpy(np.stack([s[0] for s in samples])).long().to(device)
    piece_squares, _ = piece_gather(boards[:, :BOARD_SQUARES])
    target_policy_np, legal_mask_np, active_np = _piece_targets_and_mask(
        piece_squares.cpu().numpy(), samples
    )
    target_policy = torch.from_numpy(target_policy_np).to(device)
    legal_mask = torch.from_numpy(legal_mask_np).to(device)
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
    is_rl_sample = torch.tensor([len(s) > 7 for s in samples], device=device)
    old_log_probs = torch.tensor(
        [s[7] if len(s) > 7 else 0.0 for s in samples],
        dtype=torch.float32,
        device=device,
    )
    sample_temperatures = torch.tensor(
        [s[8] if len(s) > 8 else 1.0 for s in samples],
        dtype=torch.float32,
        device=device,
    )

    ref_log_probs = None
    if ref_model is not None and kl_coef > 0:
        with torch.no_grad():
            ref_heatmap, _ = ref_model(
                boards, use_diffuser=use_diffuser, diffuser_steps=diffuser_steps
            )
            ref_log_probs = F.log_softmax(
                ref_heatmap.masked_fill(~legal_mask, -1e4), dim=-1
            ).clamp(min=-20.0)

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
        heatmaps, values = model(
            boards, use_diffuser=use_diffuser, diffuser_steps=diffuser_steps
        )
        masked = heatmaps.masked_fill(~legal_mask, -1e4)
        log_probs = F.log_softmax(masked, dim=-1).clamp(min=-20.0)
        tempered_log_probs = F.log_softmax(
            log_probs / sample_temperatures[:, None, None], dim=-1
        )
        per_piece_log_prob = (target_policy * tempered_log_probs).sum(dim=-1)
        sample_log_probs = (per_piece_log_prob * active).sum(dim=-1) / active_count

        ratio = torch.exp(sample_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        rl_term = -torch.minimum(ratio * policy_weights, clipped_ratio * policy_weights)
        sft_term = -sample_log_probs * policy_weights
        policy_loss = torch.where(is_rl_sample, rl_term, sft_term).mean()

        piece_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy = ((piece_entropy * active).sum(dim=-1) / active_count).mean()

        value_loss = (
            value_weights * F.mse_loss(values, target_values, reduction="none")
        ).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        if vae is not None:
            aux_loss = vae_loss(vae, board_images(boards).to(device), vae_kl_weight)
            loss = loss + vae_loss_weight * aux_loss
        if ref_model is not None and kl_coef > 0:
            assert ref_log_probs is not None
            piece_kl = (
                (log_probs.exp() * (log_probs - ref_log_probs))
                .masked_fill(~legal_mask, 0.0)
                .sum(dim=-1)
            )
            loss = (
                loss + kl_coef * ((piece_kl * active).sum(dim=-1) / active_count).mean()
            )

    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    with torch.no_grad():
        target_entropy = -(
            target_policy * torch.log(target_policy.clamp(min=1e-12))
        ).sum(dim=-1)
        kl_div = (
            ((-per_piece_log_prob.detach() - target_entropy) * active).sum(dim=-1)
            / active_count
        ).mean()
        top1_match = (
            target_policy.argmax(dim=-1) == log_probs.detach().argmax(dim=-1)
        ).float()
        top1_acc = ((top1_match * active).sum(dim=-1) / active_count).mean()

    if vae is not None:
        vae_opt.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        if vae is not None:
            scaler.unscale_(vae_opt)
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(vae_opt)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if vae is not None:
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            vae_opt.step()

    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        kl_div.item(),
        top1_acc.item(),
    )
