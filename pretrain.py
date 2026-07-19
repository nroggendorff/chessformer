import os

from tqdm import tqdm

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
):
    if len(replay.pretrain_buf) < config.pretrain_batch_size:
        return

    eval_interval = max(1, total_steps // config.elo_eval_count)
    pbar = tqdm(range(total_steps), desc="Pretraining Optimization")
    for step in pbar:
        batch = replay.sample_pretrain(config.pretrain_batch_size)
        loss, policy_loss, value_loss, kl_div, top1_acc = train_batch(
            train_model, opt, scaler, batch, device
        )
        scheduler.step()

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
                **elo_postfix,
            }
        )


if __name__ == "__main__":
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
    )
    save_checkpoint(model, checkpoint_path)
    save_optimizer_state(opt, scheduler, checkpoint_path)
