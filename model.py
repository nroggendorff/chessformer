import torch
import torch.nn as nn

from encoding import ACTION_SIZE, SEQ_LEN, VOCAB_SIZE


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
        self.action_emb = nn.Embedding(ACTION_SIZE, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh()
        )

    def forward(self, board_tokens, action_ids):
        B, S = board_tokens.shape
        board = self.encoder(
            self.token_emb(board_tokens)
            + self.pos_emb(torch.arange(S, device=board_tokens.device).expand(B, S))
        )
        attn_out, _ = self.cross_attn(self.action_emb(action_ids), board, board)
        policy_logits = self.policy_head(attn_out).squeeze(-1)
        value = self.value_head(board.mean(dim=1)).squeeze(-1)
        return policy_logits, value
