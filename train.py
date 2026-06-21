import os

import torch
from safetensors.torch import save_file

from config import Config
from model import ChessNet
from pretrain import run_pretraining
from replay_buffer import DualRingBuffer
from self_play import run_self_play


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model = ChessNet(
        d_model=config.d_model, nhead=config.nhead, enc_layers=config.enc_layers
    ).to(device)
    train_model = (
        torch.compile(model, mode="reduce-overhead") if device.type == "cuda" else model
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    replay = DualRingBuffer(
        pretrain_capacity=config.pretrain_capacity, rl_capacity=config.rl_capacity
    )

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    run_pretraining(train_model, opt, scaler, replay, device, config)
    run_self_play(model, train_model, opt, scaler, replay, device, config)

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(model_dir, "chessformer.safetensors"))


if __name__ == "__main__":
    main()
