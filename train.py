import os

import torch

from config import (
    Config,
    build_model,
    build_optimizer,
    build_scaler,
    build_scheduler,
    default_checkpoint_path,
    get_device,
)
from dataset import DEFAULT_PATH, generate_pretrain_dataset, load_pretrain_dataset
from model import save_checkpoint
from pretrain import run_pretraining
from replay_buffer import DualRingBuffer
from self_play import run_self_play


def main():
    config = Config()
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model, train_model = build_model(config, device)
    opt = build_optimizer(model, config)
    scaler = build_scaler(device)
    scheduler = build_scheduler(
        opt,
        config.pretrain_steps
        + config.self_play_iterations * config.self_play_gradient_steps,
    )
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    replay.extend_pretrain(
        load_pretrain_dataset(DEFAULT_PATH)
        if os.path.exists(DEFAULT_PATH)
        else generate_pretrain_dataset(config, DEFAULT_PATH)
    )

    elo_state = {}
    run_pretraining(
        model, train_model, opt, scaler, scheduler, replay, device, config, elo_state
    )
    run_self_play(
        model, train_model, opt, scaler, scheduler, replay, device, config, elo_state
    )

    save_checkpoint(model, default_checkpoint_path())


if __name__ == "__main__":
    main()
