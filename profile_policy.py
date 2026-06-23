import chess
import torch
from torch.profiler import ProfilerActivity, profile

from config import get_device
from model import ChessNet
from policy import batched_policy_step


def main():
    device = get_device()
    model = ChessNet().to(device).eval()
    boards = [chess.Board() for _ in range(128)]

    with torch.inference_mode():
        for _ in range(3):
            batched_policy_step(boards, model, device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(20):
                batched_policy_step(boards, model, device)

    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))


if __name__ == "__main__":
    main()
