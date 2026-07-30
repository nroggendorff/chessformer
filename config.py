import math
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


def cgroup_cpu_quota():
    v2_path = "/sys/fs/cgroup/cpu.max"
    if os.path.exists(v2_path):
        quota, period = open(v2_path).read().split()
        return None if quota == "max" else math.ceil(int(quota) / int(period))
    quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(quota_path) and os.path.exists(period_path):
        quota, period = int(open(quota_path).read()), int(open(period_path).read())
        return math.ceil(quota / period) if quota > 0 else None
    return None


def cgroup_memory_limit_mb():
    v2_path = "/sys/fs/cgroup/memory.max"
    if os.path.exists(v2_path):
        limit = open(v2_path).read().strip()
        if limit != "max":
            return int(limit) / (1024**2)
    v1_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(v1_path):
        limit = int(open(v1_path).read())
        if limit <= 2**62:
            return limit / (1024**2)
    meminfo_path = "/proc/meminfo"
    if os.path.exists(meminfo_path):
        for line in open(meminfo_path):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024
    return None


def physical_cpu_count():
    if not os.path.exists("/proc/cpuinfo"):
        return None
    physical_ids, current_physical = set(), None
    with open("/proc/cpuinfo") as f:
        for line in f:
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
    pretrain_chunk_games: int = 20
    pretrain_max_moves: int = 120
    pretrain_sample_moves: int = 30
    pretrain_traj_depth: int = 8
    pretrain_depth: int = 12
    pretrain_sample_multipv: int = 6
    pretrain_node_cap: int | None = 300000
    pretrain_epochs: int = 2
    pretrain_batch_size: int = 128
    pretrain_hash_mb: int = 128
    pretrain_drive_depth: int = 3
    pretrain_drive_multipv: int = 8
    pretrain_endgame_weight: float = 2.0
    pretrain_policy_temperature: float = 0.06
    pretrain_drive_temperature: float = 0.3
    pretrain_sample_ply_ramp: int = 10
    pretrain_max_sample_win_prob: float = 0.85
    pretrain_max_sample_entropy: float = 1.5
    pretrain_sample_stability: int = 2
    pretrain_sample_score_margin: int = 25

    self_play_iterations: int = 100
    self_play_games_per_iter: int = 128
    self_play_temperature: float = 1.0
    self_play_temperature_floor: float = 0.1
    self_play_max_moves: int = 150
    self_play_sample_moves: int = 15
    self_play_batch_size: int = 128
    self_play_gradient_steps: int = 16
    self_play_decisive_weight: float = 1.5
    self_play_lr: float = 2e-5
    self_play_promote_z: float = 1.0
    self_play_rollback_z: float = 2.0
    self_play_rollback_patience: int = 3
    self_play_resign_threshold: float = 0.95
    self_play_resign_streak: int = 2
    self_play_pool_size: int = 8
    self_play_pool_self_prob: float = 0.75
    self_play_pool_update_interval: int = 25
    self_play_eval_count: int = 8
    self_play_max_workers: int | None = 1
    self_play_worker_max_tasks: int = 256
    self_play_chunk_games: int = 20
    self_play_memory_safety_margin_mb: float = 3072.0

    self_play_mcts_simulations: int = 64
    self_play_opponent_mcts_simulations: int = 64
    inference_mcts_simulations: int = 400
    mcts_sims_per_wave: int = 8
    mcts_target_batch_size: int = 8192
    mcts_max_batch_size: int = 1024
    mcts_c_puct: float = 1.5
    mcts_dirichlet_alpha: float = 0.3
    mcts_root_noise_frac: float = 0.25

    elo_eval_count: int = 2
    elo_eval_games: int = 60
    elo_eval_mcts_simulations: int = 128
    elo_eval_random_plies: int = 8
    elo_eval_anchor: int = 1550
    elo_eval_max_moves: int = 120
    elo_eval_movetime: float = 0.2
    elo_eval_ema_alpha: float = 0.3

    pretrain_capacity: int = 55360000
    rl_capacity: int = 200000

    lr: float = 2e-3
    weight_decay: float = 1e-2

    d_model: int = 128
    nhead: int = 4
    enc_layers: int = 6
    heatmap_hidden: int = 32
    attn_type_rank: int = 16

    max_workers: int | None = None

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(
                physical_cpu_count() or multiprocessing.cpu_count(),
                cgroup_cpu_quota() or math.inf,
            )
        if self.self_play_max_workers is None:
            self.self_play_max_workers = self.max_workers

    def pretrain_steps_for(self, dataset_size):
        return max(1, self.pretrain_epochs * dataset_size // self.pretrain_batch_size)


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


def set_optimizer_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr
    return opt


def build_scaler(device):
    return (
        torch.amp.GradScaler(device.type)
        if device.type == "cuda" and amp_dtype(device) == torch.float16
        else None
    )


def build_scheduler(opt, total_steps):
    warmup_steps = min(500, total_steps - 1, max(20, total_steps // 20))
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
