import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dense_policy_and_mask(samples):
    target_policy = np.zeros((len(samples), 64, 64), dtype=np.float32)
    legal_mask = np.zeros((len(samples), 64, 64), dtype=bool)
    for i, (_, legal_pairs, policy_pairs, policy_probs, _) in enumerate(samples):
        legal_mask[i, legal_pairs[:, 0], legal_pairs[:, 1]] = True
        target_policy[i, policy_pairs[:, 0], policy_pairs[:, 1]] = policy_probs
    return target_policy, legal_mask


def train_batch(model, opt, scaler, samples, device):
    model.train()

    boards = torch.from_numpy(np.stack([s[0] for s in samples])).long().to(device)
    target_policy_np, legal_mask_np = _dense_policy_and_mask(samples)
    target_policy = torch.from_numpy(target_policy_np).to(device)
    legal_mask = torch.from_numpy(legal_mask_np).to(device)
    target_values = torch.tensor(
        [s[4] for s in samples], dtype=torch.float32, device=device
    )

    opt.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
    ):
        logits, value_pred = model(boards)
        logits = logits.masked_fill(~legal_mask, -1e4)
        logits_flat = logits.view(logits.size(0), -1)
        target_flat = target_policy.view(target_policy.size(0), -1)
        policy_loss = (
            -(target_flat * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1).mean()
        )
        value_loss = F.mse_loss(value_pred, target_values)
        loss = policy_loss + 0.5 * value_loss

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

    return loss.item(), policy_loss.item(), value_loss.item()
