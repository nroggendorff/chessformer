import chess

TOKEN_EMPTY = 0
BOARD_SQUARES = 64
SEQ_LEN = 68
INPUT_SIZE = SEQ_LEN + 4 * BOARD_SQUARES
VOCAB_SIZE = 48
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


def canonical_square(square, board):
    return chess.square_mirror(square) if board.turn == chess.BLACK else square


def board_to_tokens(board):
    mover, opponent = board.turn, not board.turn
    flip = mover == chess.BLACK
    tokens = [TOKEN_EMPTY] * BOARD_SQUARES
    mover_bb, opponent_bb = board.occupied_co[mover], board.occupied_co[opponent]

    for piece_type, attr in PIECE_BBS:
        bb = getattr(board, attr)
        for sq in chess.scan_reversed(bb & mover_bb):
            tokens[chess.square_mirror(sq) if flip else sq] = piece_type
        for sq in chess.scan_reversed(bb & opponent_bb):
            tokens[chess.square_mirror(sq) if flip else sq] = piece_type + 6

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
        ]
    )
    return tokens


def board_to_input(board):
    legal_froms = {canonical_square(m.from_square, board) for m in board.legal_moves}
    legal_tos = {canonical_square(m.to_square, board) for m in board.legal_moves}
    last_from, last_to = [0] * BOARD_SQUARES, [0] * BOARD_SQUARES
    if board.move_stack:
        mv = board.peek()
        last_from[canonical_square(mv.from_square, board)] = 1
        last_to[canonical_square(mv.to_square, board)] = 1
    return (
        board_to_tokens(board)
        + [int(i in legal_froms) for i in range(BOARD_SQUARES)]
        + [int(i in legal_tos) for i in range(BOARD_SQUARES)]
        + last_from
        + last_to
    )


def legal_moves_by_square_pair(board, include_promotions=True):
    moves = {}
    for move in board.legal_moves:
        if not include_promotions and move.promotion is not None:
            continue
        key = (
            canonical_square(move.from_square, board),
            canonical_square(move.to_square, board),
        )
        if move.promotion in (None, chess.QUEEN):
            moves[key] = move
        else:
            moves.setdefault(key, move)
    return moves
