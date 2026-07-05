import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from encoding import BOARD_SQUARES, NUM_PIECE_TOKENS, SEQ_LEN, VOCAB_SIZE
from piece_attention import PieceAwareEncoder, kv_color_ids, query_type_ids

META_KV_COLOR = 3


class ChessNet(nn.Module):
    def __init__(
        self, d_model=128, nhead=4, enc_layers=2, heatmap_hidden=128, attn_rank=32
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
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(d_model, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, BOARD_SQUARES),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, 1),
        )
        (
            self.legal_from_emb,
            self.legal_to_emb,
            self.last_from_emb,
            self.last_to_emb,
        ) = [nn.Embedding(2, d_model) for _ in range(4)]
        self.register_buffer(
            "ranks", torch.arange(BOARD_SQUARES) // 8, persistent=False
        )
        self.register_buffer("files", torch.arange(BOARD_SQUARES) % 8, persistent=False)
        self.register_buffer(
            "meta_positions", torch.arange(SEQ_LEN - BOARD_SQUARES), persistent=False
        )

    def forward(self, board_input):
        B = board_input.size(0)
        board_tokens, meta_tokens = (
            board_input[:, :BOARD_SQUARES],
            board_input[:, BOARD_SQUARES:SEQ_LEN],
        )
        encoded = self.encoder(
            torch.cat(
                [
                    self.token_emb(board_tokens)
                    + self.rank_emb(self.ranks.expand(B, BOARD_SQUARES))
                    + self.file_emb(self.files.expand(B, BOARD_SQUARES))
                    + self.legal_from_emb(
                        board_input[:, SEQ_LEN : SEQ_LEN + BOARD_SQUARES]
                    )
                    + self.legal_to_emb(
                        board_input[
                            :, SEQ_LEN + BOARD_SQUARES : SEQ_LEN + 2 * BOARD_SQUARES
                        ]
                    )
                    + self.last_from_emb(
                        board_input[
                            :,
                            SEQ_LEN + 2 * BOARD_SQUARES : SEQ_LEN + 3 * BOARD_SQUARES,
                        ]
                    )
                    + self.last_to_emb(board_input[:, SEQ_LEN + 3 * BOARD_SQUARES :]),
                    self.token_emb(meta_tokens)
                    + self.meta_pos_emb(
                        self.meta_positions.expand(B, SEQ_LEN - BOARD_SQUARES)
                    ),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    query_type_ids(board_tokens),
                    torch.full_like(meta_tokens, NUM_PIECE_TOKENS),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    kv_color_ids(board_tokens),
                    torch.full_like(meta_tokens, META_KV_COLOR),
                ],
                dim=1,
            ),
        )
        return (
            self.heatmap_mlp(encoded[:, :BOARD_SQUARES]),
            self.value_mlp(encoded.mean(dim=1)).squeeze(-1).tanh(),
        )


def load_checkpoint(path, device, config):
    model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
        attn_rank=config.attn_type_rank,
    ).to(device)
    model.load_state_dict(load_file(path, device="cpu"))
    model.eval()
    return model


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(model.state_dict(), path)
