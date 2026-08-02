import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from encoding import BOARD_SQUARES, SEQ_LEN, VOCAB_SIZE

MAX_PIECES = 16
NUM_RELATIONS = 15 * 15 + 1
REL_BIAS_SCALE = 4.0


def relative_position_ids():
    ranks, files = torch.arange(BOARD_SQUARES) // 8, torch.arange(BOARD_SQUARES) % 8
    board_ids = (ranks[:, None] - ranks[None, :] + 7) * 15 + (
        files[:, None] - files[None, :] + 7
    )
    ids = torch.full((SEQ_LEN, SEQ_LEN), NUM_RELATIONS - 1, dtype=torch.long)
    ids[:BOARD_SQUARES, :BOARD_SQUARES] = board_ids
    return ids


class RelativeTransformerEncoder(nn.Module):
    rel_ids: torch.Tensor

    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.nhead = nhead
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.rel_bias = nn.ParameterList(
            nn.Parameter(torch.zeros(NUM_RELATIONS, nhead)) for _ in range(num_layers)
        )
        self.register_buffer("rel_ids", relative_position_ids(), persistent=False)

    def forward(self, x):
        B, N, _ = x.shape
        for layer, bias_table in zip(self.layers, self.rel_bias):
            bias = (REL_BIAS_SCALE * torch.tanh(bias_table[self.rel_ids])).permute(
                2, 0, 1
            )
            mask = (
                bias.unsqueeze(0)
                .expand(B, self.nhead, N, N)
                .reshape(B * self.nhead, N, N)
            )
            x = layer(x, src_mask=mask)
        return x


def piece_gather(board_tokens):
    own_piece = (board_tokens >= 1) & (board_tokens <= 6)
    order = torch.argsort(own_piece.long(), dim=-1, descending=True, stable=True)
    piece_squares = order[:, :MAX_PIECES]
    piece_mask = torch.gather(own_piece, 1, piece_squares)
    return piece_squares, piece_mask


class ChessNet(nn.Module):
    ranks: torch.Tensor
    files: torch.Tensor

    def __init__(
        self,
        d_model=128,
        nhead=4,
        enc_layers=2,
        heatmap_hidden=128,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.rank_emb, self.file_emb = [nn.Embedding(n, d_model) for n in (8, 8)]
        self.encoder = RelativeTransformerEncoder(
            d_model, nhead, 4 * d_model, enc_layers
        )
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(d_model, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, BOARD_SQUARES),
        )
        self.value_key = nn.Linear(d_model, d_model)
        self.value_value = nn.Linear(d_model, d_model)
        self.value_query = nn.Parameter(torch.zeros(d_model))
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, 1),
        )
        self.legal_to_mlp = nn.Sequential(
            nn.Linear(BOARD_SQUARES, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, d_model),
        )
        self.legal_from_emb = nn.Linear(BOARD_SQUARES, d_model)
        self.register_buffer(
            "ranks", torch.arange(BOARD_SQUARES) // 8, persistent=False
        )
        self.register_buffer("files", torch.arange(BOARD_SQUARES) % 8, persistent=False)
        self.register_buffer(
            "bit_positions", torch.arange(BOARD_SQUARES), persistent=False
        )

    def forward(self, board_input, value_only=False):
        B = board_input.size(0)
        board_tokens, meta_tokens = (
            board_input[:, :BOARD_SQUARES],
            board_input[:, BOARD_SQUARES:SEQ_LEN],
        )
        legal_to_bits = board_input[:, SEQ_LEN : SEQ_LEN + BOARD_SQUARES]
        legal_from = board_input[:, SEQ_LEN + BOARD_SQUARES :].float()
        legal_to = ((legal_to_bits.unsqueeze(-1) >> self.bit_positions) & 1).float()
        legal_from_embed = self.legal_from_emb(legal_from).unsqueeze(1)
        seq = torch.cat(
            [
                self.token_emb(board_tokens)
                + self.rank_emb(self.ranks.expand(B, BOARD_SQUARES))
                + self.file_emb(self.files.expand(B, BOARD_SQUARES))
                + self.legal_to_mlp(legal_to)
                + legal_from_embed,
                self.token_emb(meta_tokens) + legal_from_embed,
            ],
            dim=1,
        )
        encoded = self.encoder(seq)
        attn = torch.softmax(
            self.value_key(encoded) @ self.value_query / self.d_model**0.5, dim=1
        )
        pooled = (attn.unsqueeze(-1) * self.value_value(encoded)).sum(dim=1)
        value = self.value_mlp(pooled).squeeze(-1).tanh()
        if value_only:
            return None, value

        piece_squares, _ = piece_gather(board_tokens)
        piece_embeds = torch.gather(
            encoded[:, :BOARD_SQUARES],
            1,
            piece_squares.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        heatmap = self.heatmap_mlp(piece_embeds)
        return heatmap, value


def load_checkpoint(path, device, config):
    model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
    ).to(device)
    model.load_state_dict(load_file(path, device="cpu"))
    model.eval()
    return model


def save_checkpoint(model, path):
    bad = [k for k, v in model.state_dict().items() if not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"Refusing to save non-finite tensors: {bad[:5]}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(model.state_dict(), path)
