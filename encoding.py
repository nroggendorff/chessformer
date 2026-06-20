import chess

PROMO_TO_ID = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
ID_TO_PROMO = {v: k for k, v in PROMO_TO_ID.items()}
ACTION_SIZE = 64 * 64 * 5

TOKEN_EMPTY = 0
VOCAB_SIZE = 40
SEQ_LEN = 67


def move_to_index(move):
    return (move.from_square * 64 + move.to_square) * 5 + PROMO_TO_ID.get(
        move.promotion, 0
    )


def index_to_move(idx):
    promo = ID_TO_PROMO[idx % 5]
    idx //= 5
    return chess.Move(idx // 64, idx % 64, promotion=promo)


PIECE_BBS = (
    (chess.PAWN, "pawns"),
    (chess.KNIGHT, "knights"),
    (chess.BISHOP, "bishops"),
    (chess.ROOK, "rooks"),
    (chess.QUEEN, "queens"),
    (chess.KING, "kings"),
)


def board_to_tokens(board):
    tokens = [TOKEN_EMPTY] * 64
    white_bb, black_bb = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]

    for piece_type, attr in PIECE_BBS:
        bb = getattr(board, attr)
        for sq in chess.scan_reversed(bb & white_bb):
            tokens[sq] = piece_type
        for sq in chess.scan_reversed(bb & black_bb):
            tokens[sq] = piece_type + 6

    castling = (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    )

    tokens.extend(
        [
            13 if board.turn == chess.WHITE else 14,
            15 + castling,
            31 if board.ep_square is None else 32 + chess.square_file(board.ep_square),
        ]
    )
    return tokens
