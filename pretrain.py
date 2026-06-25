from tqdm import tqdm

from data_generation import generate_pretrain_data
from evaluation import estimate_elo
from training import train_batch


def run_pretraining(
    model, train_model, opt, scaler, scheduler, replay, device, config, elo_state
):
    generate_pretrain_data(config, replay)
    print(f"Generated {len(replay.pretrain_buf)} pretraining positions.")

    if len(replay.pretrain_buf) < config.pretrain_batch_size:
        return

    pbar = tqdm(range(config.pretrain_steps), desc="Pretraining Optimization")
    for step in pbar:
        loss, p_loss, v_loss = train_batch(
            train_model,
            opt,
            scaler,
            replay.sample(config.pretrain_batch_size, mix_ratio=1.0),
            device,
        )
        scheduler.step()

        if (step + 1) % config.elo_eval_interval == 0:
            estimate_elo(model, device, config, elo_state)
        elo_postfix = (
            {"elo": f"{elo_state['elo_ema']:.0f}"} if "elo_ema" in elo_state else {}
        )

        pbar.set_postfix(
            loss=f"{loss:.3f}",
            policy=f"{p_loss:.3f}",
            value=f"{v_loss:.3f}",
            **elo_postfix,
        )
