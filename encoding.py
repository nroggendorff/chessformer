import pecan as pc

TOKEN_EMPTY = 0
BOARD_SIZE = pc.BOARD_SIZE
BOARD_SQUARES = pc.NUM_SQUARES
NUM_OWN_PIECE_TYPES = len(pc.PIECE_TYPES)

CLOCK_BUCKETS = 5
CLOCK_BASE = 1 + 2 * NUM_OWN_PIECE_TYPES
REPETITION_BASE = CLOCK_BASE + CLOCK_BUCKETS
STM_BASE = REPETITION_BASE + 3
VOCAB_SIZE = STM_BASE + 2

SEQ_LEN = BOARD_SQUARES + 3
INPUT_SIZE = SEQ_LEN + 2 * BOARD_SQUARES


def canon_square(sq, mover):
    return sq if mover == pc.WHITE else pc.square_mirror(sq)


def canon_bitboard(bitboard, mover):
    return bitboard if mover == pc.WHITE else pc.flip_vertical(bitboard)


def board_to_tokens(board):
    mover, opp = board.turn, not board.turn
    tokens = [TOKEN_EMPTY] * BOARD_SQUARES
    for sq in range(BOARD_SQUARES):
        piece = board.board[sq]
        if piece is None:
            continue
        color, ptype = piece
        token = ptype if color == mover else ptype + NUM_OWN_PIECE_TYPES
        tokens[canon_square(sq, mover)] = token

    repetition = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
    tokens.extend(
        [
            CLOCK_BASE + min(board.halfmove_clock // 8, CLOCK_BUCKETS - 1),
            REPETITION_BASE + repetition,
            STM_BASE + int(mover == pc.BLACK),
        ]
    )
    return tokens


def board_to_input(board):
    mover = board.turn
    last_from = [0] * BOARD_SQUARES
    if board.move_stack:
        last_from[canon_square(board.move_stack[-1].from_square, mover)] = 1
    legal_to = [0] * BOARD_SQUARES
    for origin in range(BOARD_SQUARES):
        if board.piece_at(origin):
            legal_to[canon_square(origin, mover)] = canon_bitboard(
                board.attacks(origin), mover
            )
    return board_to_tokens(board) + legal_to + last_from


def legal_moves_by_square_pair(board, legal_moves=None, include_promotions=True):
    mover = board.turn
    moves = {}
    for move in board.legal_moves if legal_moves is None else legal_moves:
        if not include_promotions and move.promotion is not None:
            continue
        key = (
            canon_square(move.from_square, mover),
            canon_square(move.to_square, mover),
        )
        if move.promotion in (None, pc.CHANCELLOR):
            moves[key] = move
        else:
            moves.setdefault(key, move)
    return moves
