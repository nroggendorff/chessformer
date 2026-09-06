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
    set_optimizer_lr,
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

    checkpoint_path = default_checkpoint_path()
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
    model, train_model = build_model(config, device, checkpoint_path)
    opt = build_optimizer(model, config)
    scaler = build_scaler(device)
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    replay.extend_pretrain(
        (
            load_pretrain_dataset(DEFAULT_PATH)
            if os.path.exists(DEFAULT_PATH)
            else generate_pretrain_dataset(config, DEFAULT_PATH)
        ),
        pool_size=config.pretrain_shuffle_pool,
        chunk_size=config.pretrain_chunk_rows,
    )
    pretrain_steps = config.pretrain_steps_for(len(replay.pretrain_buf))
    print(
        f"Training for {pretrain_steps} steps "
        f"({config.pretrain_epochs} epochs over {len(replay.pretrain_buf)} examples)"
    )

    elo_state = {}
    pretrain_scheduler = build_scheduler(opt, pretrain_steps)
    run_pretraining(
        model,
        train_model,
        opt,
        scaler,
        pretrain_scheduler,
        replay,
        device,
        config,
        elo_state,
        pretrain_steps,
        checkpoint_path=checkpoint_path,
    )
    save_checkpoint(model, checkpoint_path)

    set_optimizer_lr(opt, config.self_play_lr)
    self_play_scheduler = build_scheduler(
        opt, config.self_play_iterations * config.self_play_gradient_steps
    )
    run_self_play(
        model,
        train_model,
        opt,
        scaler,
        self_play_scheduler,
        replay,
        device,
        config,
        elo_state,
        checkpoint_path=checkpoint_path,
    )

    save_checkpoint(model, checkpoint_path)


if __name__ == "__main__":
    main()
