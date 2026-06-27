import torch
import torch.nn as nn
from safetensors.torch import load_file

from encoding import BOARD_SQUARES, SEQ_LEN, VOCAB_SIZE

PIECE_TYPES = 6

N, S, E, W = (0, 1), (0, -1), (1, 0), (-1, 0)
NE, NW, SE, SW = (1, 1), (-1, 1), (1, -1), (-1, -1)
ROOK_DIRECTIONS = [N, S, E, W]
BISHOP_DIRECTIONS = [NE, NW, SE, SW]
QUEEN_DIRECTIONS = ROOK_DIRECTIONS + BISHOP_DIRECTIONS

PAWN_OFFSETS = [(0, 1), (0, 2), (-1, 1), (1, 1)]
KNIGHT_OFFSETS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]
KING_OFFSETS = ROOK_DIRECTIONS + BISHOP_DIRECTIONS + [(2, 0), (-2, 0)]


def sliding_offsets(directions):
    return [(df * dist, dr * dist) for df, dr in directions for dist in range(1, 8)]


PIECE_OFFSETS = [
    PAWN_OFFSETS,
    KNIGHT_OFFSETS,
    sliding_offsets(BISHOP_DIRECTIONS),
    sliding_offsets(ROOK_DIRECTIONS),
    sliding_offsets(QUEEN_DIRECTIONS),
    KING_OFFSETS,
]


def build_offset_tables():
    widths = [len(offsets) for offsets in PIECE_OFFSETS]
    tables = torch.full(
        (PIECE_TYPES, BOARD_SQUARES, max(widths)), BOARD_SQUARES, dtype=torch.long
    )
    for piece_idx, offsets in enumerate(PIECE_OFFSETS):
        for square in range(BOARD_SQUARES):
            file, rank = square % 8, square // 8
            for offset_idx, (df, dr) in enumerate(offsets):
                nf, nr = file + df, rank + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    tables[piece_idx, square, offset_idx] = nr * 8 + nf
    return tables, widths


class ChessNet(nn.Module):
    def __init__(
        self, d_model=128, nhead=4, enc_layers=2, heatmap_hidden=128, move_emb_dim=32
    ):
        super().__init__()
        self.d_model = d_model
        self.move_emb_dim = move_emb_dim
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
        offset_tables, piece_widths = build_offset_tables()
        self.register_buffer("offset_tables", offset_tables, persistent=False)
        self.piece_heatmap_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, heatmap_hidden),
                nn.GELU(),
                nn.Linear(heatmap_hidden, width * move_emb_dim),
            )
            for width in piece_widths
        )
        self.move_score_head = nn.Sequential(
            nn.Linear(move_emb_dim, heatmap_hidden),
            nn.GELU(),
            nn.Linear(heatmap_hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh()
        )
        self.register_buffer("positions", torch.arange(SEQ_LEN), persistent=False)

    def _piece_heatmap(self, mlp, table, squares):
        move_embs = mlp(squares).view(*squares.shape[:2], -1, self.move_emb_dim)
        scores = self.move_score_head(move_embs).squeeze(-1)
        idx = table[:, : scores.size(-1)].unsqueeze(0).expand(scores.size(0), -1, -1)
        scratch = torch.full(
            (scores.size(0), BOARD_SQUARES, BOARD_SQUARES + 1),
            -1e4,
            device=scores.device,
            dtype=scores.dtype,
        )
        return scratch.scatter_(-1, idx, scores)[..., :BOARD_SQUARES]

    def forward(self, board_tokens):
        B, S = board_tokens.shape
        board = self.encoder(
            self.token_emb(board_tokens) + self.pos_emb(self.positions[:S].expand(B, S))
        )
        squares = board[:, :BOARD_SQUARES]
        heatmaps = torch.stack(
            [
                self._piece_heatmap(mlp, self.offset_tables[i], squares)
                for i, mlp in enumerate(self.piece_heatmap_mlps)
            ],
            dim=2,
        )
        piece_idx = (board_tokens[:, :BOARD_SQUARES] - 1).clamp(0, PIECE_TYPES - 1)
        from_heatmaps = heatmaps.gather(
            2, piece_idx[:, :, None, None].expand(-1, -1, 1, BOARD_SQUARES)
        ).squeeze(2)
        value = self.value_head(board.mean(dim=1)).squeeze(-1)
        return from_heatmaps, value


def load_checkpoint(path, device, config):
    model = ChessNet(
        d_model=config.d_model,
        nhead=config.nhead,
        enc_layers=config.enc_layers,
        heatmap_hidden=config.heatmap_hidden,
        move_emb_dim=config.move_emb_dim,
    ).to(device)
    model.load_state_dict(load_file(path, device="cpu"))
    model.eval()
    return model
