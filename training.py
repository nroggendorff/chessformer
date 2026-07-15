import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import amp_dtype
from policy import joint_move_log_probs


def _dense_policy_and_mask(samples):
    target_policy = np.zeros((len(samples), 64, 64), dtype=np.float32)
    legal_mask = np.zeros((len(samples), 64, 64), dtype=bool)
    for i, (_, legal_pairs, policy_pairs, policy_probs, *_) in enumerate(samples):
        legal_mask[i, legal_pairs[:, 0], legal_pairs[:, 1]] = True
        target_policy[i, policy_pairs[:, 0], policy_pairs[:, 1]] = policy_probs
    return target_policy, legal_mask


def train_batch(
    model, opt, scaler, samples, device, ref_model=None, kl_coef=0.0, clip_epsilon=0.2
):
    model.train()

    boards = torch.from_numpy(np.stack([s[0] for s in samples])).long().to(device)
    target_policy_np, legal_mask_np = _dense_policy_and_mask(samples)
    target_policy = torch.from_numpy(target_policy_np).to(device)
    legal_mask = torch.from_numpy(legal_mask_np).to(device)
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
            ref_log_probs = joint_move_log_probs(
                ref_model(boards)[0], legal_mask
            ).clamp(min=-20.0)

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
        heatmaps, values = model(boards)
        log_probs = joint_move_log_probs(heatmaps, legal_mask).clamp(min=-20.0)
        flat_log_probs = log_probs.view(log_probs.size(0), -1)
        flat_target = target_policy.view(target_policy.size(0), -1)
        tempered_log_probs = F.log_softmax(
            flat_log_probs / sample_temperatures[:, None], dim=-1
        )
        sample_log_probs = (flat_target * tempered_log_probs).sum(dim=-1)

        ratio = torch.exp(sample_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        rl_term = -torch.minimum(ratio * policy_weights, clipped_ratio * policy_weights)
        sft_term = -sample_log_probs * policy_weights
        policy_loss = torch.where(is_rl_sample, rl_term, sft_term).mean()

        entropy = -(flat_log_probs.exp() * flat_log_probs).sum(dim=-1).mean()
        value_loss = (
            value_weights * F.mse_loss(values, target_values, reduction="none")
        ).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        if ref_model is not None and kl_coef > 0:
            assert ref_log_probs is not None
            loss = (
                loss
                + kl_coef
                * (log_probs.exp() * (log_probs - ref_log_probs))
                .masked_fill(~legal_mask, 0.0)
                .sum(dim=(-2, -1))
                .mean()
            )

    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    with torch.no_grad():
        target_entropy = -(flat_target * torch.log(flat_target.clamp(min=1e-12))).sum(
            dim=-1
        )
        kl_div = (-sample_log_probs.detach() - target_entropy).mean()
        top1_acc = (
            (flat_target.argmax(dim=-1) == flat_log_probs.detach().argmax(dim=-1))
            .float()
            .mean()
        )

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
