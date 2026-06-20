import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_batch(model, opt, scaler, samples, device):
    model.train()

    boards = torch.from_numpy(np.stack([s[0] for s in samples])).long().to(device)
    target_policy = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
    target_values = torch.tensor(
        [s[2] for s in samples], dtype=torch.float32, device=device
    )

    opt.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
    ):
        logits, value_pred = model(boards)
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
