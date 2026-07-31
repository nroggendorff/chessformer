import chess

TOKEN_EMPTY = 0
BOARD_SQUARES = 64
SEQ_LEN = 69
INPUT_SIZE = SEQ_LEN + 2 * BOARD_SQUARES
VOCAB_SIZE = 50
NUM_PIECE_TOKENS = 13

PIECE_BBS = (
    (chess.PAWN, "pawns"),
    (chess.KNIGHT, "knights"),
    (chess.BISHOP, "bishops"),
    (chess.ROOK, "rooks"),
    (chess.QUEEN, "queens"),
    (chess.KING, "kings"),
)

CASTLING_BASE = 13
EP_NONE = 29
EP_FILE_BASE = 30
CLOCK_BASE = 38
REPETITION_BASE = 45
STM_BASE = 48


def board_to_tokens(board):
    mover, opponent = board.turn, not board.turn
    tokens = [TOKEN_EMPTY] * BOARD_SQUARES
    mover_bb, opponent_bb = board.occupied_co[mover], board.occupied_co[opponent]

    for piece_type, attr in PIECE_BBS:
        bb = getattr(board, attr)
        for sq in chess.scan_reversed(bb & mover_bb):
            tokens[sq] = piece_type
        for sq in chess.scan_reversed(bb & opponent_bb):
            tokens[sq] = piece_type + 6

    castling = (
        int(board.has_kingside_castling_rights(mover))
        | int(board.has_queenside_castling_rights(mover)) << 1
        | int(board.has_kingside_castling_rights(opponent)) << 2
        | int(board.has_queenside_castling_rights(opponent)) << 3
    )
    ep_token = (
        EP_NONE
        if board.ep_square is None
        else EP_FILE_BASE + chess.square_file(board.ep_square)
    )
    repetition = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0

    tokens.extend(
        [
            CASTLING_BASE + castling,
            ep_token,
            CLOCK_BASE + min(board.halfmove_clock // 10, 6),
            REPETITION_BASE + repetition,
            STM_BASE + int(mover == chess.BLACK),
        ]
    )
    return tokens


def _as_int64(bitboard):
    return bitboard - (1 << 64) if bitboard >= (1 << 63) else bitboard


def board_to_input(board):
    last_from = [0] * BOARD_SQUARES
    if board.move_stack:
        last_from[board.peek().from_square] = 1
    legal_to = [
        _as_int64(int(board.attacks(origin))) if board.piece_at(origin) else 0
        for origin in chess.SQUARES
    ]
    return board_to_tokens(board) + legal_to + last_from


def legal_moves_by_square_pair(board, legal_moves=None, include_promotions=True):
    moves = {}
    for move in board.legal_moves if legal_moves is None else legal_moves:
        if not include_promotions and move.promotion is not None:
            continue
        key = (move.from_square, move.to_square)
        if move.promotion in (None, chess.QUEEN):
            moves[key] = move
        else:
            moves.setdefault(key, move)
    return moves
