import os

from tqdm import tqdm

from diffuser import diffuser_train_step
from evaluation import estimate_elo
from training import train_batch


def run_pretraining(
    model,
    train_model,
    opt,
    scaler,
    scheduler,
    replay,
    device,
    config,
    elo_state,
    total_steps,
    vae=None,
    vae_opt=None,
    diffuser=None,
    diffuser_opt=None,
    diffuser_scheduler=None,
):
    if len(replay.pretrain_buf) < config.pretrain_batch_size:
        return

    fuse_diffuser = diffuser is not None and config.diffuser_fusion_enabled

    eval_interval = max(1, total_steps // config.elo_eval_count)
    pbar = tqdm(range(total_steps), desc="Pretraining Optimization")
    for step in pbar:
        batch = replay.sample_pretrain(config.pretrain_batch_size)
        loss, policy_loss, value_loss, kl_div, top1_acc = train_batch(
            train_model,
            opt,
            scaler,
            batch,
            device,
            vae=vae,
            vae_opt=vae_opt,
            vae_kl_weight=config.vae_kl_weight,
            vae_loss_weight=config.vae_loss_weight,
            use_diffuser=fuse_diffuser,
            diffuser_steps=config.diffuser_inference_steps,
        )
        scheduler.step()

        diffuser_postfix = {}
        if diffuser is not None:
            diffuser_loss = diffuser_train_step(
                vae,
                diffuser,
                diffuser_scheduler,
                diffuser_opt,
                [s[0] for s in batch],
                [s[2] for s in batch],
                [s[3] for s in batch],
                device,
            )
            diffuser_postfix = {"diffuser": f"{diffuser_loss:.3f}"}

        if (step + 1) % eval_interval == 0:
            estimate_elo(model, device, config, elo_state)
            pbar.unpause()
        elo_postfix = (
            {"elo": f"{elo_state['elo_ema']:.0f}"} if "elo_ema" in elo_state else {}
        )

        pbar.set_postfix(
            {
                "loss": f"{loss:.3f}",
                "policy": f"{policy_loss:.3f}",
                "value": f"{value_loss:.3f}",
                "kl": f"{kl_div:.3f}",
                "top1": f"{top1_acc:.1%}",
                **diffuser_postfix,
                **elo_postfix,
            }
        )


if __name__ == "__main__":
    import torch

    from config import (
        Config,
        build_model,
        build_optimizer,
        build_scaler,
        build_scheduler,
        default_checkpoint_path,
        get_device,
        load_optimizer_state,
        optimizer_state_path,
        save_optimizer_state,
    )
    from dataset import DEFAULT_PATH, generate_pretrain_dataset, load_pretrain_dataset
    from model import save_checkpoint
    from replay_buffer import DualRingBuffer

    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()
    resuming = os.path.exists(checkpoint_path)
    print(
        f"Resuming pretraining from {checkpoint_path}"
        if resuming
        else "Starting pretraining from scratch"
    )
    model, train_model = build_model(config, device, checkpoint_path)
    opt = build_optimizer(model, config)
    scaler = build_scaler(device)
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )
    replay.extend_pretrain(
        load_pretrain_dataset(DEFAULT_PATH)
        if os.path.exists(DEFAULT_PATH)
        else generate_pretrain_dataset(config, DEFAULT_PATH)
    )
    total_steps = config.pretrain_steps_for(len(replay.pretrain_buf))
    print(
        f"Training for {total_steps} steps ({config.pretrain_epochs} epochs over {len(replay.pretrain_buf)} examples)"
    )
    scheduler = build_scheduler(opt, total_steps)
    if resuming and os.path.exists(optimizer_state_path(checkpoint_path)):
        load_optimizer_state(opt, scheduler, checkpoint_path)
        print("Resumed optimizer and LR schedule state from prior run")
    elif resuming:
        print(
            "No saved optimizer state found — this run will restart the LR warmup "
            "and Adam moments against an already-trained checkpoint, which can "
            "degrade it. Consider restoring from a backup instead."
        )

    vae_opt = torch.optim.AdamW(model.vae.parameters(), lr=config.vae_lr)
    diffuser_opt = (
        torch.optim.AdamW(model.diffuser.parameters(), lr=config.diffuser_lr)
        if config.diffuser_enabled
        else None
    )

    run_pretraining(
        model,
        train_model,
        opt,
        scaler,
        scheduler,
        replay,
        device,
        config,
        {},
        total_steps,
        vae=model.vae,
        vae_opt=vae_opt,
        diffuser=model.diffuser if config.diffuser_enabled else None,
        diffuser_opt=diffuser_opt,
        diffuser_scheduler=model.diffuser_scheduler,
    )
    save_checkpoint(model, checkpoint_path)
    save_optimizer_state(opt, scheduler, checkpoint_path)
