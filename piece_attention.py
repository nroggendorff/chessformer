import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import NUM_PIECE_TOKENS

NUM_QUERY_TYPES = NUM_PIECE_TOKENS + 1
NUM_KV_COLORS = 4


def query_type_ids(tokens):
    return torch.clamp(tokens, max=NUM_PIECE_TOKENS)


def kv_color_ids(tokens):
    return torch.where(
        (tokens >= 1) & (tokens <= 6),
        torch.zeros_like(tokens),
        torch.where(
            (tokens >= 7) & (tokens <= 12),
            torch.ones_like(tokens),
            torch.full_like(tokens, 2),
        ),
    )


def init_lowrank_bank(num, d_in, d_out, rank):
    down = nn.Parameter(torch.empty(num, d_in, rank))
    [nn.init.kaiming_uniform_(w, a=5**0.5) for w in down]
    return down, nn.Parameter(torch.zeros(num, rank, d_out))


def type_conditioned_linear(x, type_ids, weight):
    T, d_in, d_out = weight.shape
    projected = (x @ weight.permute(1, 0, 2).reshape(d_in, T * d_out)).view(
        *x.shape[:-1], T, d_out
    )
    return torch.gather(
        projected, -2, type_ids[..., None, None].expand(*type_ids.shape, 1, d_out)
    ).squeeze(-2)


def type_conditioned_delta(x, type_ids, down, up):
    return type_conditioned_linear(
        type_conditioned_linear(x, type_ids, down), type_ids, up
    )


class PieceAwareAttention(nn.Module):
    def __init__(self, d_model, nhead, rank=32):
        super().__init__()
        self.nhead, self.head_dim = nhead, d_model // nhead
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(d_model, d_model) for _ in range(3)
        )
        self.q_down, self.q_up = init_lowrank_bank(
            NUM_QUERY_TYPES, d_model, d_model, rank
        )
        self.k_down, self.k_up = init_lowrank_bank(
            NUM_KV_COLORS, d_model, d_model, rank
        )
        self.v_down, self.v_up = init_lowrank_bank(
            NUM_KV_COLORS, d_model, d_model, rank
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, query_types, kv_colors):
        B, N, D = x.shape
        q, k, v = (
            (proj(x) + type_conditioned_delta(x, ids, down, up))
            .view(B, N, self.nhead, self.head_dim)
            .transpose(1, 2)
            for proj, ids, down, up in (
                (self.q_proj, query_types, self.q_down, self.q_up),
                (self.k_proj, kv_colors, self.k_down, self.k_up),
                (self.v_proj, kv_colors, self.v_down, self.v_up),
            )
        )
        return self.out_proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        )


class PieceAwareEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, rank=32):
        super().__init__()
        self.attn = PieceAwareAttention(d_model, nhead, rank)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x, query_types, kv_colors):
        x = x + self.attn(self.norm1(x), query_types, kv_colors)
        return x + self.ff(self.norm2(x))


class PieceAwareEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, rank=32):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PieceAwareEncoderLayer(d_model, nhead, dim_feedforward, rank)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, query_types, kv_colors):
        for layer in self.layers:
            x = layer(x, query_types, kv_colors)
        return x
