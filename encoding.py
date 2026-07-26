import chess
import numpy as np

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


def board_to_input(board):
    last_from = [0] * BOARD_SQUARES
    if board.move_stack:
        last_from[board.peek().from_square] = 1
    attacked = {
        square
        for origin in chess.SQUARES
        if board.piece_at(origin)
        for square in board.attacks(origin)
    }
    return (
        board_to_tokens(board)
        + [int(sq in attacked) for sq in range(BOARD_SQUARES)]
        + last_from
    )


def board_square_tokens(board):
    return board_to_tokens(board)[:BOARD_SQUARES]


def child_board(board, move):
    child = board.copy()
    child.push(move)
    return child


def move_touched_squares(board, move):
    squares = {move.from_square, move.to_square}
    if board.is_en_passant(move):
        squares.add(move.to_square + (-8 if board.turn == chess.WHITE else 8))
    elif board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(
            move.from_square
        )
        squares.update(
            {
                chess.square(7 if kingside else 0, rank),
                chess.square(5 if kingside else 3, rank),
            }
        )
    return squares


def board_state_target(board, move_weights):
    total = sum(move_weights.values()) or 1.0
    tokens = board_square_tokens(board)
    base = [0 if t == 0 else t + 6 if t <= 6 else t - 6 for t in tokens]

    dist = {}
    for move, weight in move_weights.items():
        weight /= total
        child_tokens = board_square_tokens(child_board(board, move))
        for square in move_touched_squares(board, move):
            votes = dist.setdefault(square, {})
            votes[child_tokens[square]] = votes.get(child_tokens[square], 0.0) + weight

    for square, votes in dist.items():
        covered = sum(votes.values())
        if covered < 1.0:
            votes[base[square]] = votes.get(base[square], 0.0) + (1.0 - covered)

    flat = [
        (square, token, weight)
        for square, votes in dist.items()
        for token, weight in votes.items()
    ]
    if not flat:
        return (
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.float32),
        )
    squares, tokens_, weights = zip(*flat)
    return (
        np.array(squares, dtype=np.uint8),
        np.array(tokens_, dtype=np.uint8),
        np.array(weights, dtype=np.float32),
    )
