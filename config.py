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

    pretrain_games: int = 130000
    pretrain_games_per_task: int = 100
    pretrain_max_moves: int = 50
    pretrain_sample_moves: int = 20
    pretrain_traj_depth: int = 3
    pretrain_depth: int = 8
    pretrain_samples_target: int = 20000000
    pretrain_batch_size: int = 1024
    pretrain_hash_mb: int = 128
    pretrain_multipv: int = 10

    self_play_iterations: int = 2000
    self_play_games_per_iter: int = 32
    self_play_temperature: float = 1.0
    self_play_max_moves: int = 120
    self_play_sample_moves: int = 15
    self_play_batch_size: int = 256
    self_play_gradient_steps: int = 6
    self_play_mix_ratio: float = 0.5
    self_play_td_lambda: float = 0.8
    self_play_adv_clip: float = 3.0

    elo_eval_count: int = 8
    elo_eval_games: int = 12
    elo_eval_anchor: int = 1320
    elo_eval_max_moves: int = 60
    elo_eval_depth: int = 1
    elo_eval_ema_alpha: float = 0.3

    pretrain_capacity: int = 55360000
    rl_capacity: int = 200000

    lr: float = 3e-4
    weight_decay: float = 1e-2

    d_model: int = 256
    nhead: int = 8
    enc_layers: int = 8
    heatmap_hidden: int = 256

    max_workers: int = None

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = multiprocessing.cpu_count()

    @property
    def pretrain_steps(self):
        return max(1, self.pretrain_samples_target // self.pretrain_batch_size)
