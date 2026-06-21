import torch
import torch.nn as nn

from encoding import BOARD_SQUARES, NUM_PIECE_TOKENS, SEQ_LEN, VOCAB_SIZE


class ChessNet(nn.Module):
    def __init__(self, d_model=128, nhead=4, enc_layers=2):
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
        self.from_proj = nn.Embedding(NUM_PIECE_TOKENS, d_model * d_model)
        nn.init.normal_(self.from_proj.weight, std=d_model**-0.5)
        self.to_proj = nn.Linear(d_model, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh()
        )
        self.scale = d_model**-0.5
        self.register_buffer("positions", torch.arange(SEQ_LEN), persistent=False)

    def forward(self, board_tokens):
        B, S = board_tokens.shape
        board = self.encoder(
            self.token_emb(board_tokens) + self.pos_emb(self.positions[:S].expand(B, S))
        )
        squares = board[:, :BOARD_SQUARES]
        from_weights = self.from_proj(board_tokens[:, :BOARD_SQUARES]).view(
            B, BOARD_SQUARES, self.d_model, self.d_model
        )
        from_q = torch.einsum("bid,bidf->bif", squares, from_weights)
        policy_logits = (
            torch.einsum("bid,bjd->bij", from_q, self.to_proj(squares)) * self.scale
        )
        value = self.value_head(board.mean(dim=1)).squeeze(-1)
        return policy_logits, value
