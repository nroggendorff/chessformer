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
        (
            self.legal_from_emb,
            self.legal_to_emb,
            self.last_from_emb,
            self.last_to_emb,
        ) = [nn.Embedding(2, d_model) for _ in range(4)]
        self.register_buffer("positions", torch.arange(SEQ_LEN), persistent=False)

    def forward(self, board_input):
        B = board_input.size(0)
        x = self.token_emb(board_input[:, :SEQ_LEN]) + self.pos_emb(
            self.positions.expand(B, SEQ_LEN)
        )
        x = torch.cat(
            [
                x[:, :BOARD_SQUARES]
                + self.legal_from_emb(board_input[:, SEQ_LEN : SEQ_LEN + BOARD_SQUARES])
                + self.legal_to_emb(
                    board_input[
                        :, SEQ_LEN + BOARD_SQUARES : SEQ_LEN + 2 * BOARD_SQUARES
                    ]
                )
                + self.last_from_emb(
                    board_input[
                        :, SEQ_LEN + 2 * BOARD_SQUARES : SEQ_LEN + 3 * BOARD_SQUARES
                    ]
                )
                + self.last_to_emb(board_input[:, SEQ_LEN + 3 * BOARD_SQUARES :]),
                x[:, BOARD_SQUARES:],
            ],
            dim=1,
        )
        return self.heatmap_mlp(self.encoder(x)[:, :BOARD_SQUARES])


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
