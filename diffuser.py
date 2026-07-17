import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler

from encoding import BOARD_SQUARES

VAE_CHANNELS = 13
VAE_LATENT_CHANNELS = 4
VAE_BLOCK_CHANNELS = (32, 64)
VAE_NORM_GROUPS = 8


def build_vae():
    return AutoencoderKL(
        in_channels=VAE_CHANNELS,
        out_channels=VAE_CHANNELS,
        down_block_types=("DownEncoderBlock2D",) * len(VAE_BLOCK_CHANNELS),
        up_block_types=("UpDecoderBlock2D",) * len(VAE_BLOCK_CHANNELS),
        block_out_channels=VAE_BLOCK_CHANNELS,
        layers_per_block=1,
        latent_channels=VAE_LATENT_CHANNELS,
        norm_num_groups=VAE_NORM_GROUPS,
        sample_size=8,
    )


@torch.no_grad()
def infer_latent_dim(vae, device="cpu"):
    return (
        vae.encode(torch.zeros(1, VAE_CHANNELS, 8, 8, device=device))
        .latent_dist.sample()
        .numel()
    )


class DiffuserNet(nn.Module):
    def __init__(self, latent_dim, hidden=256, depth=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.cond_proj = nn.Linear(latent_dim, hidden)
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(depth)
        )
        self.out_proj = nn.Linear(hidden, latent_dim)

    def forward(self, noisy_latent, timestep, cond_latent):
        h = (
            self.in_proj(noisy_latent)
            + self.cond_proj(cond_latent)
            + self.time_emb(timestep.view(-1, 1).float())
        )
        for block in self.blocks:
            h = h + block(h)
        return self.out_proj(h)


def build_noise_scheduler(num_train_timesteps=100):
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)


def tokens_to_image(tokens):
    return (
        F.one_hot(tokens.long(), VAE_CHANNELS)
        .permute(0, 2, 1)
        .float()
        .reshape(-1, VAE_CHANNELS, 8, 8)
    )


def board_images(board_inputs):
    if isinstance(board_inputs, (list, tuple)):
        board_inputs = np.array(board_inputs)
    return tokens_to_image(torch.as_tensor(board_inputs)[:, :BOARD_SQUARES])


def desirability_targets(board_inputs, policy_pairs_list, policy_probs_list):
    board_inputs = torch.as_tensor(board_inputs)
    targets = torch.zeros(len(board_inputs), VAE_CHANNELS, BOARD_SQUARES)
    for i, (pairs, probs) in enumerate(zip(policy_pairs_list, policy_probs_list)):
        pairs = torch.as_tensor(pairs).long()
        piece_types = board_inputs[i, pairs[:, 0]].long()
        targets[i, piece_types, pairs[:, 1]] = torch.as_tensor(
            probs, dtype=torch.float32
        )
    return targets.reshape(-1, VAE_CHANNELS, 8, 8)


def vae_loss(vae, images, kl_weight=1e-6):
    posterior = vae.encode(images).latent_dist
    recon = vae.decode(posterior.sample()).sample
    return F.mse_loss(recon, images) + kl_weight * posterior.kl().mean()


def diffuser_train_step(
    vae,
    diffuser,
    scheduler,
    opt,
    board_inputs,
    policy_pairs_list,
    policy_probs_list,
    device,
):
    with torch.no_grad():
        cond_latent = vae.encode(
            board_images(board_inputs).to(device)
        ).latent_dist.mode()
        target_latent = vae.encode(
            desirability_targets(board_inputs, policy_pairs_list, policy_probs_list).to(
                device
            )
        ).latent_dist.mode()

    diffuser.train()
    noise = torch.randn_like(target_latent)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (target_latent.size(0),), device=device
    )
    noisy_latent = scheduler.add_noise(target_latent, noise, timesteps)
    pred_noise = diffuser(
        noisy_latent.flatten(1), timesteps, cond_latent.flatten(1)
    ).view_as(noise)
    loss = F.mse_loss(pred_noise, noise)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(diffuser.parameters(), 1.0)
    opt.step()
    return loss.item()
