import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import MAX_LEGAL_MOVES


def train_batch(model, opt, scaler, samples, device):
    model.train()
    batch_size = len(samples)

    boards = torch.from_numpy(np.stack([s[0] for s in samples])).long().to(device)

    action_ids_np = np.zeros((batch_size, MAX_LEGAL_MOVES), dtype=np.int64)
    target_policy_np = np.zeros((batch_size, MAX_LEGAL_MOVES), dtype=np.float32)
    mask_np = np.zeros((batch_size, MAX_LEGAL_MOVES), dtype=bool)

    for i, (_, a_ids, probs, _) in enumerate(samples):
        m = len(a_ids)
        action_ids_np[i, :m] = a_ids
        target_policy_np[i, :m] = probs
        mask_np[i, :m] = True

    action_ids = torch.from_numpy(action_ids_np).to(device)
    target_policy = torch.from_numpy(target_policy_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    target_values = torch.tensor(
        [s[3] for s in samples], dtype=torch.float32, device=device
    )

    opt.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
    ):
        logits, value_pred = model(boards, action_ids)
        logits = logits.masked_fill(~mask, -1e4)
        policy_loss = (
            -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
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
