import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import NUM_PIECE_TOKENS

NUM_QUERY_TYPES = NUM_PIECE_TOKENS + 1
NUM_KV_COLORS = 3


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


def init_weight_bank(num, d_model):
    weight = torch.empty(num, d_model, d_model)
    [nn.init.xavier_uniform_(w) for w in weight]
    return nn.Parameter(weight)


def type_conditioned_linear(x, type_ids, weight):
    B, N, D = x.shape
    flat_x, flat_types = x.reshape(-1, D), type_ids.reshape(-1)
    out = torch.zeros_like(flat_x)
    for t in range(weight.shape[0]):
        mask = flat_types == t
        if mask.any():
            out[mask] = (flat_x[mask] @ weight[t]).to(out.dtype)
    return out.view(B, N, D)


class PieceAwareAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead, self.head_dim = nhead, d_model // nhead
        self.q_weight = init_weight_bank(NUM_QUERY_TYPES, d_model)
        self.k_weight = init_weight_bank(NUM_KV_COLORS, d_model)
        self.v_weight = init_weight_bank(NUM_KV_COLORS, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, query_types, kv_colors):
        B, N, D = x.shape
        q, k, v = (
            type_conditioned_linear(x, ids, weight)
            .view(B, N, self.nhead, self.head_dim)
            .transpose(1, 2)
            for ids, weight in (
                (query_types, self.q_weight),
                (kv_colors, self.k_weight),
                (kv_colors, self.v_weight),
            )
        )
        return self.out_proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        )


class PieceAwareEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.attn = PieceAwareAttention(d_model, nhead)
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
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PieceAwareEncoderLayer(d_model, nhead, dim_feedforward)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, query_types, kv_colors):
        for layer in self.layers:
            x = layer(x, query_types, kv_colors)
        return x
