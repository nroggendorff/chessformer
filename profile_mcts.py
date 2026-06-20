import torch
from torch.profiler import ProfilerActivity, profile

from mcts import MCTSNode, batched_mcts_sim
from model import ChessNet
import chess


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device).eval()
    if device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

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

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
