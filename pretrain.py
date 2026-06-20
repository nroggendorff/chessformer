from tqdm import tqdm

from data_generation import generate_pretrain_data
from training import train_batch


def run_pretraining(
    model,
    opt,
    scaler,
    replay,
    device,
    stockfish_path,
    pretrain_games=15000,
    batch_size=512,
    steps=12000,
):
    replay.extend_pretrain(generate_pretrain_data(stockfish_path, pretrain_games))
    print(f"Generated {len(replay.pretrain_buf)} pretraining positions.")

    if len(replay.pretrain_buf) < batch_size:
        return

    pbar = tqdm(range(steps), desc="Pretraining Optimization")
    for _ in pbar:
        loss, p_loss, v_loss = train_batch(
            model, opt, scaler, replay.sample(batch_size, mix_ratio=1.0), device
        )
        pbar.set_postfix(
            loss=f"{loss:.3f}", policy=f"{p_loss:.3f}", value=f"{v_loss:.3f}"
        )
