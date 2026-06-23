import torch
import torch.nn as nn

from encoding import BOARD_SQUARES, SEQ_LEN, VOCAB_SIZE

PIECE_TYPES = 6


class ChessNet(nn.Module):
    def __init__(self, d_model=128, nhead=4, enc_layers=2, heatmap_hidden=128):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=enc_layers,
        )
        self.piece_heatmap_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, heatmap_hidden),
                nn.GELU(),
                nn.Linear(heatmap_hidden, BOARD_SQUARES),
            )
            for _ in range(PIECE_TYPES)
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh()
        )
        self.register_buffer("positions", torch.arange(SEQ_LEN), persistent=False)

    def forward(self, board_tokens):
        B, S = board_tokens.shape
        board = self.encoder(
            self.token_emb(board_tokens) + self.pos_emb(self.positions[:S].expand(B, S))
        )
        squares = board[:, :BOARD_SQUARES]
        heatmaps = torch.stack([mlp(squares) for mlp in self.piece_heatmap_mlps], dim=2)
        piece_idx = (board_tokens[:, :BOARD_SQUARES] - 1).clamp(0, PIECE_TYPES - 1)
        from_heatmaps = heatmaps.gather(
            2, piece_idx[:, :, None, None].expand(-1, -1, 1, BOARD_SQUARES)
        ).squeeze(2)
        value = self.value_head(board.mean(dim=1)).squeeze(-1)
        return from_heatmaps, value
