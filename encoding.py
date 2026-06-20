import chess

TOKEN_EMPTY = 0
VOCAB_SIZE = 40
SEQ_LEN = 67
BOARD_SQUARES = 64

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


def legal_moves_by_square_pair(board):
    moves = {}
    for move in board.legal_moves:
        if move.promotion in (None, chess.QUEEN):
            moves[(move.from_square, move.to_square)] = move
        else:
            moves.setdefault((move.from_square, move.to_square), move)
    return moves
