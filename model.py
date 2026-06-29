import torch
import torch.nn as nn
from safetensors.torch import load_file

from encoding import BOARD_SQUARES, SEQ_LEN, VOCAB_SIZE


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
        self.register_buffer("positions", torch.arange(SEQ_LEN), persistent=False)

    def forward(self, board_tokens):
        B, S = board_tokens.shape
        board = self.encoder(
            self.token_emb(board_tokens) + self.pos_emb(self.positions[:S].expand(B, S))
        )
        heatmap = self.heatmap_mlp(board[:, :BOARD_SQUARES])
        value = self.value_mlp(board.mean(dim=1)).squeeze(-1).tanh()
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
