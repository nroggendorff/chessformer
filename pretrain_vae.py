import os

import torch
from tqdm import tqdm

from config import Config, build_model, default_checkpoint_path, get_device
from dataset import DEFAULT_PATH, generate_pretrain_dataset, load_pretrain_dataset
from diffuser import board_images, vae_loss
from model import save_checkpoint

EPOCHS = 3
BATCH_SIZE = 512


def run_vae_pretraining(vae, opt, dataset, device, epochs, batch_size, kl_weight):
    steps_per_epoch = max(1, len(dataset) // batch_size)
    pbar = tqdm(range(epochs * steps_per_epoch), desc="VAE Pretraining")
    vae.train()
    for step in pbar:
        rows = dataset.select(torch.randint(0, len(dataset), (batch_size,)).tolist())[
            "board_input"
        ]
        loss = vae_loss(vae, board_images(rows).to(device), kl_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})


if __name__ == "__main__":
    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()

    dataset = (
        load_pretrain_dataset(DEFAULT_PATH)
        if os.path.exists(DEFAULT_PATH)
        else generate_pretrain_dataset(config, DEFAULT_PATH)
    )
    print(f"Pretraining VAE on {len(dataset)} board positions")

    model, _ = build_model(config, device, checkpoint_path)
    opt = torch.optim.AdamW(model.vae.parameters(), lr=config.vae_lr)

    run_vae_pretraining(
        model.vae, opt, dataset, device, EPOCHS, BATCH_SIZE, config.vae_kl_weight
    )

    save_checkpoint(model, checkpoint_path)
    print(f"Saved checkpoint (with pretrained VAE) to {checkpoint_path}")
