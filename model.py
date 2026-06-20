import torch
import torch.nn as nn

from encoding import BOARD_SQUARES, SEQ_LEN, VOCAB_SIZE


class ChessNet(nn.Module):
    def __init__(self, d_model=128, nhead=4, enc_layers=2):
        super().__init__()
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
        self.from_proj = nn.Linear(d_model, d_model)
        self.to_proj = nn.Linear(d_model, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh()
        )
        self.scale = d_model**-0.5

    def forward(self, board_tokens):
        B, S = board_tokens.shape
        board = self.encoder(
            self.token_emb(board_tokens)
            + self.pos_emb(torch.arange(S, device=board_tokens.device).expand(B, S))
        )
        squares = board[:, :BOARD_SQUARES]
        policy_logits = (
            torch.einsum("bid,bjd->bij", self.from_proj(squares), self.to_proj(squares))
            * self.scale
        )
        value = self.value_head(board.mean(dim=1)).squeeze(-1)
        return policy_logits, value
