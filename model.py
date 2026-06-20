import torch
import torch.nn as nn

from encoding import ACTION_SIZE, SEQ_LEN, VOCAB_SIZE

N_PIECES = 12


class PieceAttention(nn.Module):
    def __init__(self, d_model, n_pieces=N_PIECES, d_head=32):
        super().__init__()
        self.n_pieces, self.d_head = n_pieces, d_head
        self.piece_q = nn.Parameter(torch.randn(n_pieces, d_head) * d_head**-0.5)
        self.k_proj = nn.Linear(d_model, n_pieces * d_head)
        self.v_proj = nn.Linear(d_model, n_pieces * d_head)
        self.out_proj = nn.Linear(d_head, d_model)

    def forward(self, board, board_tokens):
        B, S, _ = board.shape
        k = self.k_proj(board).view(B, S, self.n_pieces, self.d_head)
        v = self.v_proj(board).view(B, S, self.n_pieces, self.d_head)
        attn = torch.softmax(
            torch.einsum("pd,bspd->bsp", self.piece_q, k) / self.d_head**0.5, dim=1
        )
        piece_out = self.out_proj(torch.einsum("bsp,bspd->bpd", attn, v))

        piece_id = (board_tokens - 1).clamp(0, self.n_pieces - 1)
        gathered = piece_out.gather(
            1, piece_id.unsqueeze(-1).expand(-1, -1, piece_out.size(-1))
        )
        occupied = ((board_tokens >= 1) & (board_tokens <= self.n_pieces)).unsqueeze(-1)
        return gathered * occupied


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
        self.piece_attn = PieceAttention(d_model)
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
        pos = torch.arange(S, device=board_tokens.device).expand(B, S)
        board = self.encoder(self.token_emb(board_tokens) + self.pos_emb(pos))
        board = board + self.piece_attn(board, board_tokens)
        actions = self.action_emb(action_ids)

        attn_out, _ = self.cross_attn(actions, board, board)
        policy_logits = self.policy_head(attn_out).squeeze(-1)
        value = self.value_head(board.mean(dim=1)).squeeze(-1)
        return policy_logits, value
