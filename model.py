import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from encoding import BOARD_SQUARES, NUM_PIECE_TOKENS, SEQ_LEN, VOCAB_SIZE
from piece_attention import PieceAwareEncoder, kv_color_ids, query_type_ids

META_KV_COLOR = 3


class ChessNet(nn.Module):
    ranks: torch.Tensor
    files: torch.Tensor
    meta_positions: torch.Tensor

    def __init__(
        self,
        d_model=128,
        nhead=4,
        enc_layers=2,
        value_hidden=128,
        attn_rank=32,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.rank_emb, self.file_emb, self.meta_pos_emb = [
            nn.Embedding(n, d_model) for n in (8, 8, SEQ_LEN - BOARD_SQUARES)
        ]
        self.encoder = PieceAwareEncoder(
            d_model, nhead, 4 * d_model, enc_layers, attn_rank
        )
        self.board_norm = nn.LayerNorm(d_model)
        self.value_key = nn.Linear(d_model, d_model)
        self.value_value = nn.Linear(d_model, d_model)
        self.value_query = nn.Parameter(torch.zeros(d_model))
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )
        self.legal_to_emb, self.last_from_emb = [
            nn.Embedding(2, d_model) for _ in range(2)
        ]
        self.register_buffer(
            "ranks", torch.arange(BOARD_SQUARES) // 8, persistent=False
        )
        self.register_buffer("files", torch.arange(BOARD_SQUARES) % 8, persistent=False)
        self.register_buffer(
            "meta_positions", torch.arange(SEQ_LEN - BOARD_SQUARES), persistent=False
        )

    def forward(self, board_input, value_only=False):
        B = board_input.size(0)
        board_tokens, meta_tokens = (
            board_input[:, :BOARD_SQUARES],
            board_input[:, BOARD_SQUARES:SEQ_LEN],
        )
        seq = torch.cat(
            [
                self.token_emb(board_tokens)
                + self.rank_emb(self.ranks.expand(B, BOARD_SQUARES))
                + self.file_emb(self.files.expand(B, BOARD_SQUARES))
                + self.legal_to_emb(board_input[:, SEQ_LEN : SEQ_LEN + BOARD_SQUARES])
                + self.last_from_emb(board_input[:, SEQ_LEN + BOARD_SQUARES :]),
                self.token_emb(meta_tokens)
                + self.meta_pos_emb(
                    self.meta_positions.expand(B, SEQ_LEN - BOARD_SQUARES)
                ),
            ],
            dim=1,
        )
        query_types = torch.cat(
            [
                query_type_ids(board_tokens),
                torch.full_like(meta_tokens, NUM_PIECE_TOKENS),
            ],
            dim=1,
        )
        kv_colors = torch.cat(
            [kv_color_ids(board_tokens), torch.full_like(meta_tokens, META_KV_COLOR)],
            dim=1,
        )

        encoded = self.encoder(seq, query_types, kv_colors)
        attn = torch.softmax(
            self.value_key(encoded) @ self.value_query / self.d_model**0.5, dim=1
        )
        pooled = (attn.unsqueeze(-1) * self.value_value(encoded)).sum(dim=1)
        value = self.value_mlp(pooled).squeeze(-1).tanh()
        if value_only:
            return None, value

        board_logits = (
            self.board_norm(encoded[:, :BOARD_SQUARES])
            @ self.token_emb.weight[:NUM_PIECE_TOKENS].T
        )
        return board_logits, value


def load_checkpoint(path, device, config):
    model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        value_hidden=config.value_hidden,
        attn_rank=config.attn_type_rank,
    ).to(device)
    model.load_state_dict(load_file(path, device="cpu"))
    model.eval()
    return model


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(model.state_dict(), path)
