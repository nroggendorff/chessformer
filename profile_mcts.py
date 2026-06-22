import torch
from torch.profiler import ProfilerActivity, profile

from config import get_device
from mcts import MCTSNode, batched_mcts_sim
from model import ChessNet
import chess


def main():
    device = get_device()
    model = ChessNet().to(device).eval()

    boards = [chess.Board() for _ in range(128)]
    roots = [MCTSNode(prior=1.0) for _ in boards]

    with torch.inference_mode():
        for _ in range(3):
            batched_mcts_sim(roots, boards, model, device, add_noise=False)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(20):
                batched_mcts_sim(roots, boards, model, device, add_noise=False)

    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))


if __name__ == "__main__":
    main()
