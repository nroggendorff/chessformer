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


def physical_cpu_count():
    if not os.path.exists("/proc/cpuinfo"):
        return None
    physical_ids, current_physical = set(), None
    for line in open("/proc/cpuinfo"):
        if line.startswith("physical id"):
            current_physical = line.split(":")[1].strip()
        elif line.startswith("core id") and current_physical is not None:
            physical_ids.add((current_physical, line.split(":")[1].strip()))
    return len(physical_ids) or None


@dataclass
class Config:
    stockfish_path: str = field(
        default_factory=lambda: shutil.which("stockfish") or "/usr/games/stockfish"
    )

    pretrain_games: int = 125000
    pretrain_max_moves: int = 120
    pretrain_sample_moves: int = 30
    pretrain_traj_depth: int = 8
    pretrain_depth: int = 16
    pretrain_sample_multipv: int = 12
    pretrain_node_cap: int | None = 1000000
    pretrain_epochs: int = 2
    pretrain_batch_size: int = 128
    pretrain_hash_mb: int = 512
    pretrain_drive_depth: int = 3
    pretrain_drive_multipv: int = 8
    pretrain_endgame_weight: float = 2.0
    pretrain_policy_temperature: float = 0.06
    pretrain_drive_temperature: float = 0.3
    pretrain_min_sample_ply: int = 10
    pretrain_max_sample_win_prob: float = 0.85
    pretrain_min_sample_entropy: float = 0.3
    pretrain_sample_stability: int = 3
    pretrain_sample_score_margin: int = 25

    self_play_iterations: int = 400
    self_play_games_per_iter: int = 128
    self_play_temperature: float = 1.0
    self_play_temperature_floor: float = 0.25
    self_play_max_moves: int = 150
    self_play_sample_moves: int = 15
    self_play_batch_size: int = 128
    self_play_gradient_steps: int = 16
    self_play_adv_clip: float = 2.0
    self_play_draw_value: float = -0.15
    self_play_quick_win_bonus: float = 0.35
    self_play_decisive_weight: float = 1.5
    self_play_return_clip: float = 1.0
    self_play_promote_z: float = 0.5
    self_play_rollback_z: float = 1.5
    self_play_rollback_patience: int = 2
    self_play_kl_coef: float = 0.05
    self_play_clip_ratio: float = 0.2
    self_play_pool_size: int = 8
    self_play_pool_self_prob: float = 0.2
    self_play_pool_update_interval: int = 25
    self_play_ref_sync_interval: int = 50

    elo_eval_count: int = 2
    elo_eval_games: int = 90
    elo_eval_anchor: int = 1320
    elo_eval_max_moves: int = 120
    elo_eval_movetime: float = 0.2
    elo_eval_ema_alpha: float = 0.3

    pretrain_capacity: int = 55360000
    rl_capacity: int = 200000

    lr: float = 4e-4
    weight_decay: float = 1e-2

    d_model: int = 64
    nhead: int = 4
    enc_layers: int = 6
    heatmap_hidden: int = 32
    attn_type_rank: int = 16

    vae_lr: float = 1e-4
    vae_kl_weight: float = 1e-6
    vae_loss_weight: float = 0.1

    diffuser_enabled: bool = True
    diffuser_fusion_enabled: bool = True
    diffuser_hidden: int = 256
    diffuser_depth: int = 4
    diffuser_lr: float = 3e-4
    diffuser_train_timesteps: int = 100
    diffuser_inference_steps: int = 8

    max_workers: int | None = None

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = physical_cpu_count() or multiprocessing.cpu_count()

    def pretrain_steps_for(self, dataset_size):
        return max(1, self.pretrain_epochs * dataset_size // self.pretrain_batch_size)


def build_model(config, device, checkpoint_path=None):
    model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
        attn_rank=config.attn_type_rank,
        diffuser_hidden=config.diffuser_hidden,
        diffuser_depth=config.diffuser_depth,
        diffuser_train_timesteps=config.diffuser_train_timesteps,
        diffuser_inference_steps=config.diffuser_inference_steps,
        diffuser_fusion_enabled=config.diffuser_fusion_enabled,
    ).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(load_file(checkpoint_path, device="cpu"))
    train_model = (
        torch.compile(model, mode="reduce-overhead") if device.type == "cuda" else model
    )
    return model, train_model


def build_optimizer(model, config):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith(("vae.", "diffuser.")):
            continue
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


def optimizer_state_path(checkpoint_path):
    return checkpoint_path + ".opt.pt"


def save_optimizer_state(opt, scheduler, checkpoint_path):
    torch.save(
        {"optimizer": opt.state_dict(), "scheduler": scheduler.state_dict()},
        optimizer_state_path(checkpoint_path),
    )


def load_optimizer_state(opt, scheduler, checkpoint_path):
    state = torch.load(optimizer_state_path(checkpoint_path), map_location="cpu")
    opt.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
