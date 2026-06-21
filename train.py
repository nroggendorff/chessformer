import os
import shutil

import torch

from model import ChessNet
from pretrain import run_pretraining
from replay_buffer import DualRingBuffer
from self_play import run_self_play


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model = ChessNet().to(device)
    train_model = (
        torch.compile(model, mode="reduce-overhead") if device.type == "cuda" else model
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    replay = DualRingBuffer()

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    stockfish_path = shutil.which("stockfish") or "/usr/games/stockfish"

    run_pretraining(train_model, opt, scaler, replay, device, stockfish_path)
    run_self_play(model, train_model, opt, scaler, replay, device)

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "chessformer.pt"))


if __name__ == "__main__":
    main()
