import os

from tqdm import tqdm

from evaluation import estimate_elo
from training import train_batch


def run_pretraining(
    model, train_model, opt, scaler, scheduler, replay, device, config, elo_state
):
    if len(replay.pretrain_buf) < config.pretrain_batch_size:
        return

    eval_interval = max(1, config.pretrain_steps // config.elo_eval_count)
    pbar = tqdm(range(config.pretrain_steps), desc="Pretraining Optimization")
    for step in pbar:
        loss, q_loss, entropy = train_batch(
            train_model,
            opt,
            scaler,
            replay.sample(config.pretrain_batch_size, mix_ratio=1.0),
            device,
        )
        scheduler.step()

        if (step + 1) % eval_interval == 0:
            estimate_elo(model, device, config, elo_state)
            pbar.unpause()
        elo_postfix = (
            {"elo": f"{elo_state['elo_ema']:.0f}"} if "elo_ema" in elo_state else {}
        )

        pbar.set_postfix(
            loss=f"{loss:.3f}",
            q=f"{q_loss:.3f}",
            ent=f"{entropy:.3f}",
            **elo_postfix,
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
    )
    from dataset import (
        DEFAULT_PATH,
        dataset_to_samples,
        generate_pretrain_dataset,
        load_pretrain_dataset,
    )
    from model import save_checkpoint
    from replay_buffer import DualRingBuffer

    config = Config()
    device = get_device()
    checkpoint_path = default_checkpoint_path()
    print(
        f"Resuming pretraining from {checkpoint_path}"
        if os.path.exists(checkpoint_path)
        else "Starting pretraining from scratch"
    )
    model, train_model = build_model(config, device, checkpoint_path)
    opt = build_optimizer(model, config)
    scaler = build_scaler(device)
    scheduler = build_scheduler(opt, config.pretrain_steps)
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )
    replay.extend_pretrain(
        dataset_to_samples(
            load_pretrain_dataset(DEFAULT_PATH)
            if os.path.exists(DEFAULT_PATH)
            else generate_pretrain_dataset(config, DEFAULT_PATH)
        )
    )

    run_pretraining(
        model, train_model, opt, scaler, scheduler, replay, device, config, {}
    )
    save_checkpoint(model, checkpoint_path)
