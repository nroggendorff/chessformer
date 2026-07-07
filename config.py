import multiprocessing
import os
import shutil
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file

from model import ChessNet


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def amp_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device.type == "mps" else torch.bfloat16


def default_checkpoint_path():
    return os.path.join(
        os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), "chessformer.safetensors"
    )


@dataclass
class Config:
    stockfish_path: str = field(
        default_factory=lambda: shutil.which("stockfish") or "/usr/games/stockfish"
    )

    pretrain_games: int = 130000
    pretrain_games_per_task: int = 10
    pretrain_max_moves: int = 120
    pretrain_sample_moves: int = 20
    pretrain_traj_depth: int = 3
    pretrain_depth: int = 8
    pretrain_samples_target: int = 20000000
    pretrain_batch_size: int = 512
    pretrain_hash_mb: int = 128
    pretrain_drive_depth: int = 3
    pretrain_drive_multipv: int = 8
    pretrain_endgame_weight: float = 2.0

    self_play_iterations: int = 1000
    self_play_games_per_iter: int = 128
    self_play_temperature: float = 1.0
    self_play_temperature_floor: float = 0.25
    self_play_max_moves: int = 150
    self_play_sample_moves: int = 15
    self_play_batch_size: int = 128
    self_play_gradient_steps: int = 32
    self_play_mix_ratio: float = 0.1
    self_play_adv_clip: float = 2.0
    self_play_draw_value: float = -0.15
    self_play_truncation_value: float = -0.35
    self_play_quick_win_bonus: float = 0.35
    self_play_decisive_weight: float = 1.5
    self_play_return_clip: float = 1.0
    self_play_rollback_margin: float = 100.0
    self_play_kl_coef: float = 0.05

    elo_eval_count: int = 8
    elo_eval_games: int = 24
    elo_eval_anchor: int = 1320
    elo_eval_max_moves: int = 100
    elo_eval_movetime: float = 0.2
    elo_eval_ema_alpha: float = 0.3

    pretrain_capacity: int = 55360000
    rl_capacity: int = 200000

    lr: float = 3e-4
    weight_decay: float = 1e-2

    d_model: int = 384
    nhead: int = 8
    enc_layers: int = 12
    heatmap_hidden: int = 384
    attn_type_rank: int = 32

    max_workers: int | None = None

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = multiprocessing.cpu_count()

    @property
    def pretrain_steps(self):
        return max(1, self.pretrain_samples_target // self.pretrain_batch_size)


def build_model(config, device, checkpoint_path=None):
    model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
        attn_rank=config.attn_type_rank,
    ).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(load_file(checkpoint_path, device="cpu"))
    train_model = (
        torch.compile(model, mode="reduce-overhead") if device.type == "cuda" else model
    )
    return model, train_model


def build_optimizer(model, config):
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (no_decay if param.ndim < 2 else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
    )


def build_scaler(device):
    return (
        torch.amp.GradScaler(device.type)
        if device.type == "cuda" and amp_dtype(device) == torch.float16
        else None
    )


def build_scheduler(opt, total_steps):
    warmup_steps = min(500, total_steps // 20)
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-2, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )
