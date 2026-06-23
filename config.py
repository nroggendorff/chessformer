import multiprocessing
import shutil
from dataclasses import dataclass, field

import torch


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


@dataclass
class Config:
    stockfish_path: str = field(
        default_factory=lambda: shutil.which("stockfish") or "/usr/games/stockfish"
    )

    pretrain_games: int = 15000
    pretrain_games_per_task: int = 50
    pretrain_max_moves: int = 60
    pretrain_depth: int = 8
    pretrain_samples_target: int = 3000 * 512
    pretrain_batch_size: int = 512

    self_play_iterations: int = 30
    self_play_games_per_iter: int = 32
    self_play_temperature: float = 1.0
    self_play_max_moves: int = 60
    self_play_sample_moves: int = 15
    self_play_batch_size: int = 512
    self_play_gradient_steps: int = 2
    self_play_mix_ratio: float = 0.5

    pretrain_capacity: int = 500000
    rl_capacity: int = 100000

    lr: float = 3e-4
    weight_decay: float = 1e-4

    d_model: int = 128
    nhead: int = 4
    enc_layers: int = 2
    heatmap_hidden: int = 128

    max_workers: int = None

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = multiprocessing.cpu_count()

    @property
    def pretrain_steps(self):
        return max(1, self.pretrain_samples_target // self.pretrain_batch_size)
