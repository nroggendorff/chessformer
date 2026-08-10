import random
import time
from typing import NamedTuple, Optional

BOARD_SIZE = 6
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
FILES = "abcdef"

WHITE = True
BLACK = False
COLORS = (WHITE, BLACK)

PAWN, KNIGHT, SENTINEL, BISHOP, ROOK, CHANCELLOR, KING = range(1, 8)
PIECE_TYPES = (PAWN, KNIGHT, SENTINEL, BISHOP, ROOK, CHANCELLOR, KING)
PROMOTION_PIECE_TYPES = (CHANCELLOR, ROOK, BISHOP, KNIGHT, SENTINEL)

PIECE_SYMBOLS = {
    PAWN: "P",
    KNIGHT: "N",
    SENTINEL: "S",
    BISHOP: "B",
    ROOK: "R",
    CHANCELLOR: "C",
    KING: "K",
}
SYMBOL_TO_PIECE = {v: k for k, v in PIECE_SYMBOLS.items()}

VALUES = {
    PAWN: 100,
    KNIGHT: 310,
    SENTINEL: 300,
    BISHOP: 330,
    ROOK: 500,
    CHANCELLOR: 880,
    KING: 0,
}

ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_OFFS = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
KING_OFFS = ROOK_DIRS + BISHOP_DIRS
SENTINEL_OFFS = ((2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (2, -2), (-2, 2), (-2, -2))

BACK_RANK = (ROOK, KNIGHT, KING, CHANCELLOR, BISHOP, SENTINEL)

NON_KING_STARTING_MATERIAL = (
    2 * sum(VALUES[t] for t in BACK_RANK if t != KING) + 2 * 6 * VALUES[PAWN]
)


def in_bounds(f, r):
    return 0 <= f < BOARD_SIZE and 0 <= r < BOARD_SIZE


def square(file, rank):
    return rank * BOARD_SIZE + file


def square_file(sq):
    return sq % BOARD_SIZE


def square_rank(sq):
    return sq // BOARD_SIZE


def square_name(sq):
    return FILES[square_file(sq)] + str(square_rank(sq) + 1)


def parse_square(name):
    return square(FILES.index(name[0]), int(name[1:]) - 1)


def square_mirror(sq):
    return square(square_file(sq), BOARD_SIZE - 1 - square_rank(sq))


def opponent(color):
    return not color


class Move(NamedTuple):
    from_square: int
    to_square: int
    promotion: Optional[int] = None

    def uci(self):
        promo = PIECE_SYMBOLS[self.promotion].lower() if self.promotion else ""
        return square_name(self.from_square) + square_name(self.to_square) + promo

    @classmethod
    def from_uci(cls, text):
        frm, to = parse_square(text[:2]), parse_square(text[2:4])
        promo = SYMBOL_TO_PIECE[text[4].upper()] if len(text) > 4 else None
        return cls(frm, to, promo)


def _slide_targets(board, f, r, dirs, color):
    out = []
    for df, dr in dirs:
        nf, nr = f + df, r + dr
        while in_bounds(nf, nr):
            occ = board[square(nf, nr)]
            if occ is None:
                out.append(square(nf, nr))
            else:
                if occ[0] != color:
                    out.append(square(nf, nr))
                break
            nf, nr = nf + df, nr + dr
    return out


def _leap_targets(board, f, r, offs, color):
    out = []
    for df, dr in offs:
        nf, nr = f + df, r + dr
        if not in_bounds(nf, nr):
            continue
        occ = board[square(nf, nr)]
        if occ is None or occ[0] != color:
            out.append(square(nf, nr))
    return out


def _pawn_targets(board, f, r, color):
    d = 1 if color == WHITE else -1
    out = []
    if in_bounds(f, r + d) and board[square(f, r + d)] is None:
        out.append(square(f, r + d))
    for df in (-1, 1):
        nf = f + df
        if in_bounds(nf, r + d):
            occ = board[square(nf, r + d)]
            if occ is not None and occ[0] != color:
                out.append(square(nf, r + d))
    return out


def _pawn_attacks(f, r, color):
    d = 1 if color == WHITE else -1
    return [square(f + df, r + d) for df in (-1, 1) if in_bounds(f + df, r + d)]


def pseudo_targets(board, sq):
    piece = board[sq]
    color, ptype = piece
    f, r = square_file(sq), square_rank(sq)
    if ptype == PAWN:
        return _pawn_targets(board, f, r, color)
    if ptype == KNIGHT:
        return _leap_targets(board, f, r, KNIGHT_OFFS, color)
    if ptype == SENTINEL:
        return _leap_targets(board, f, r, SENTINEL_OFFS, color)
    if ptype == BISHOP:
        return _slide_targets(board, f, r, BISHOP_DIRS, color)
    if ptype == ROOK:
        return _slide_targets(board, f, r, ROOK_DIRS, color)
    if ptype == CHANCELLOR:
        return _slide_targets(board, f, r, ROOK_DIRS, color) + _leap_targets(
            board, f, r, KNIGHT_OFFS, color
        )
    return _leap_targets(board, f, r, KING_OFFS, color)


def piece_moves_from(board, sq):
    piece = board[sq]
    ptype = piece[1]
    moves = []
    if ptype == PAWN:
        f, r = square_file(sq), square_rank(sq)
        for to in _pawn_targets(board, f, r, piece[0]):
            if square_rank(to) in (0, BOARD_SIZE - 1):
                moves.extend(Move(sq, to, promo) for promo in PROMOTION_PIECE_TYPES)
            else:
                moves.append(Move(sq, to))
    else:
        moves.extend(Move(sq, to) for to in pseudo_targets(board, sq))
    return moves


def all_pseudo_moves(board, color):
    moves = []
    for sq in range(NUM_SQUARES):
        piece = board[sq]
        if piece is not None and piece[0] == color:
            moves.extend(piece_moves_from(board, sq))
    return moves


def king_square(board, color):
    for sq in range(NUM_SQUARES):
        piece = board[sq]
        if piece is not None and piece[0] == color and piece[1] == KING:
            return sq
    return -1


def is_square_attacked(board, sq, color):
    f, r = square_file(sq), square_rank(sq)
    ar = r - (1 if color == WHITE else -1)
    for df in (-1, 1):
        af = f + df
        if in_bounds(af, ar) and board[square(af, ar)] == (color, PAWN):
            return True
    for df, dr in KNIGHT_OFFS:
        nf, nr = f + df, r + dr
        if in_bounds(nf, nr):
            occ = board[square(nf, nr)]
            if occ is not None and occ[0] == color and occ[1] in (KNIGHT, CHANCELLOR):
                return True
    for df, dr in SENTINEL_OFFS:
        nf, nr = f + df, r + dr
        if in_bounds(nf, nr) and board[square(nf, nr)] == (color, SENTINEL):
            return True
    for df, dr in KING_OFFS:
        nf, nr = f + df, r + dr
        if in_bounds(nf, nr) and board[square(nf, nr)] == (color, KING):
            return True
    for df, dr in ROOK_DIRS:
        nf, nr = f + df, r + dr
        while in_bounds(nf, nr):
            occ = board[square(nf, nr)]
            if occ is not None:
                if occ[0] == color and occ[1] in (ROOK, CHANCELLOR):
                    return True
                break
            nf, nr = nf + df, nr + dr
    for df, dr in BISHOP_DIRS:
        nf, nr = f + df, r + dr
        while in_bounds(nf, nr):
            occ = board[square(nf, nr)]
            if occ is not None:
                if occ[0] == color and occ[1] == BISHOP:
                    return True
                break
            nf, nr = nf + df, nr + dr
    return False


def in_check(board, color):
    return is_square_attacked(board, king_square(board, color), opponent(color))


def make_move(board, move):
    undo = (board[move.to_square], board[move.from_square])
    moved_color = undo[1][0]
    board[move.to_square] = (moved_color, move.promotion) if move.promotion else undo[1]
    board[move.from_square] = None
    return undo


def unmake_move(board, move, undo):
    captured, moved = undo
    board[move.from_square] = moved
    board[move.to_square] = captured


def legal_moves(board, color):
    out = []
    for mv in all_pseudo_moves(board, color):
        undo = make_move(board, mv)
        if not in_check(board, color):
            out.append(mv)
        unmake_move(board, mv, undo)
    return out


def legal_captures(board, color):
    out = []
    for mv in all_pseudo_moves(board, color):
        if board[mv.to_square] is None:
            continue
        undo = make_move(board, mv)
        if not in_check(board, color):
            out.append(mv)
        unmake_move(board, mv, undo)
    return out


def initial_board():
    board = [None] * NUM_SQUARES
    for f in range(BOARD_SIZE):
        board[square(f, 0)] = (WHITE, BACK_RANK[f])
        board[square(f, 1)] = (WHITE, PAWN)
        board[square(f, BOARD_SIZE - 2)] = (BLACK, PAWN)
        board[square(f, BOARD_SIZE - 1)] = (BLACK, BACK_RANK[f])
    return board


NO_PROGRESS_LIMIT = 40


class Outcome:
    def __init__(self, winner, termination):
        self.winner = winner
        self.termination = termination


def position_status(board, color, history_hashes, no_progress):
    if not legal_moves(board, color):
        if in_check(board, color):
            return Outcome(opponent(color), "checkmate")
        return Outcome(None, "stalemate")
    if no_progress >= NO_PROGRESS_LIMIT:
        return Outcome(None, "no_progress")
    h = zobrist_hash(board, color)
    if history_hashes.count(h) >= 2:
        return Outcome(None, "repetition")
    if all(p is None or p[1] == KING for p in board):
        return Outcome(None, "insufficient")
    return None


_PIECE_CODES = [(c, t) for c in COLORS for t in PIECE_TYPES]
_PIECE_INDEX = {pc: i for i, pc in enumerate(_PIECE_CODES)}
_rng = random.Random(1804289383)
ZOBRIST = [
    [_rng.getrandbits(64) for _ in range(len(_PIECE_CODES))] for _ in range(NUM_SQUARES)
]
ZOBRIST_SIDE = _rng.getrandbits(64)


def zobrist_hash(board, color):
    h = ZOBRIST_SIDE if color == BLACK else 0
    for sq in range(NUM_SQUARES):
        piece = board[sq]
        if piece is not None:
            h ^= ZOBRIST[sq][_PIECE_INDEX[piece]]
    return h


def flip_vertical(bitboard):
    out = 0
    for sq in range(NUM_SQUARES):
        if bitboard & (1 << sq):
            out |= 1 << square_mirror(sq)
    return out


CENTER_MID = (BOARD_SIZE - 1) / 2


def _center_table(scale):
    table = [0] * NUM_SQUARES
    for r in range(BOARD_SIZE):
        for f in range(BOARD_SIZE):
            table[square(f, r)] = round(
                (CENTER_MID - (abs(f - CENTER_MID) + abs(r - CENTER_MID))) * scale
            )
    return table


CENTER_MINOR = _center_table(4)
CENTER_MAJOR = _center_table(2)
PAWN_ADV = [square_rank(sq) * 8 for sq in range(NUM_SQUARES)]
KING_MID = [(BOARD_SIZE - 1 - square_rank(sq)) * 6 for sq in range(NUM_SQUARES)]
KING_END = _center_table(6)
MAX_NON_PAWN = 2 * sum(VALUES[t] for t in (KNIGHT, SENTINEL, BISHOP, ROOK, CHANCELLOR))


def _pst_lookup(table, sq, color):
    return table[sq] if color == WHITE else table[square_mirror(sq)]


def _phase(board):
    total = sum(
        VALUES[p[1]] for p in board if p is not None and p[1] not in (KING, PAWN)
    )
    return max(0.0, min(1.0, total / MAX_NON_PAWN))


def evaluate(board):
    score = 0
    ph = _phase(board)
    for sq in range(NUM_SQUARES):
        piece = board[sq]
        if piece is None:
            continue
        color, ptype = piece
        sign = 1 if color == WHITE else -1
        score += sign * VALUES[ptype]
        if ptype == PAWN:
            score += sign * _pst_lookup(PAWN_ADV, sq, color)
        elif ptype in (KNIGHT, SENTINEL, BISHOP):
            score += sign * _pst_lookup(CENTER_MINOR, sq, color)
        elif ptype == CHANCELLOR:
            score += sign * _pst_lookup(CENTER_MAJOR, sq, color)
        elif ptype == ROOK:
            score += sign * round(_pst_lookup(CENTER_MAJOR, sq, color) * 0.5)
        elif ptype == KING:
            score += sign * round(
                _pst_lookup(KING_MID, sq, color) * ph
                + _pst_lookup(KING_END, sq, color) * (1 - ph)
            )
    return score


class TimeUp(Exception):
    pass


class Engine:
    def search(
        self, board, color, history_hashes, max_depth=6, node_cap=None, time_limit=None
    ):
        board = board[:]
        tt = {}
        killers = {}
        history_heur = {}
        state = {"nodes": 0}
        deadline = None if time_limit is None else time.monotonic() + time_limit

        def budget_ok():
            state["nodes"] += 1
            if node_cap is not None and state["nodes"] > node_cap:
                raise TimeUp()
            if (
                deadline is not None
                and state["nodes"] % 1024 == 0
                and time.monotonic() > deadline
            ):
                raise TimeUp()

        def order(moves, tt_move, ply):
            def key(mv):
                if mv == tt_move:
                    return 1_000_000
                captured = board[mv.to_square]
                if captured is not None:
                    return (
                        100_000
                        + 10 * VALUES[captured[1]]
                        - VALUES[board[mv.from_square][1]]
                    )
                score = 50_000 if mv in killers.get(ply, ()) else 0
                return score + history_heur.get(mv, 0)

            return sorted(moves, key=key, reverse=True)

        def quiescence(color, alpha, beta):
            budget_ok()
            stand_pat = evaluate(board) * (1 if color == WHITE else -1)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            for mv in order(legal_captures(board, color), None, 0):
                undo = make_move(board, mv)
                try:
                    score = -quiescence(opponent(color), -beta, -alpha)
                finally:
                    unmake_move(board, mv, undo)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha

        def negamax(color, depth, alpha, beta, ply, counts, no_progress):
            budget_ok()
            in_chk = in_check(board, color)
            moves = legal_moves(board, color)
            if not moves:
                return -100_000 + ply if in_chk else 0
            if no_progress >= NO_PROGRESS_LIMIT:
                return 0
            h = zobrist_hash(board, color)
            counts[h] = counts.get(h, 0) + 1
            if counts[h] >= 3:
                counts[h] -= 1
                return 0
            if depth <= 0:
                counts[h] -= 1
                return quiescence(color, alpha, beta)

            entry = tt.get(h)
            tt_move = None
            if entry is not None:
                tt_move = entry["move"]
                if entry["depth"] >= depth:
                    if entry["flag"] == "exact":
                        counts[h] -= 1
                        return entry["score"]
                    if entry["flag"] == "lower" and entry["score"] > alpha:
                        alpha = entry["score"]
                    elif entry["flag"] == "upper" and entry["score"] < beta:
                        beta = entry["score"]
                    if alpha >= beta:
                        counts[h] -= 1
                        return entry["score"]

            alpha_orig = alpha
            best_score, best_move = float("-inf"), None
            ext = 1 if (in_chk and ply < 24) else 0
            for mv in order(moves, tt_move, ply):
                undo = make_move(board, mv)
                captured = undo[0] is not None
                moved_pawn = undo[1][1] == PAWN
                child_no_progress = 0 if (captured or moved_pawn) else no_progress + 1
                try:
                    score = -negamax(
                        opponent(color),
                        depth - 1 + ext,
                        -beta,
                        -alpha,
                        ply + 1,
                        counts,
                        child_no_progress,
                    )
                finally:
                    unmake_move(board, mv, undo)
                if score > best_score:
                    best_score, best_move = score, mv
                if best_score > alpha:
                    alpha = best_score
                if alpha >= beta:
                    if not captured:
                        bucket = killers.setdefault(ply, [])
                        if best_move not in bucket:
                            bucket.insert(0, best_move)
                            del bucket[2:]
                        history_heur[best_move] = (
                            history_heur.get(best_move, 0) + depth * depth
                        )
                    break
            flag = (
                "upper"
                if best_score <= alpha_orig
                else ("lower" if best_score >= beta else "exact")
            )
            tt[h] = {
                "depth": depth,
                "score": best_score,
                "flag": flag,
                "move": best_move,
            }
            counts[h] -= 1
            return best_score

        counts = {h: history_hashes.count(h) for h in set(history_hashes)}
        best_move, best_score, reached_depth, root_scores = None, 0, 0, []
        root_moves = legal_moves(board, color)
        if len(root_moves) == 1:
            return {
                "move": root_moves[0],
                "score": 0,
                "depth": 0,
                "nodes": 0,
                "root_scores": [(root_moves[0], 0)],
            }

        for depth in range(1, max_depth + 1):
            local_best_move, local_best_score = None, float("-inf")
            alpha, beta = float("-inf"), float("inf")
            local_scores = []
            try:
                for mv in order(root_moves, best_move, 0):
                    undo = make_move(board, mv)
                    try:
                        score = -negamax(
                            opponent(color), depth - 1, -beta, -alpha, 1, counts, 0
                        )
                    finally:
                        unmake_move(board, mv, undo)
                    local_scores.append((mv, score))
                    if score > local_best_score:
                        local_best_score, local_best_move = score, mv
                    if local_best_score > alpha:
                        alpha = local_best_score
            except TimeUp:
                break
            if local_best_move is not None:
                best_move, best_score, reached_depth, root_scores = (
                    local_best_move,
                    local_best_score,
                    depth,
                    local_scores,
                )
            if deadline is not None and time.monotonic() > deadline:
                break
            if abs(best_score) > 90_000:
                break

        return {
            "move": best_move,
            "score": best_score,
            "depth": reached_depth,
            "nodes": state["nodes"],
            "root_scores": root_scores,
        }


class Board:
    def __init__(self, fen=None):
        if fen is None:
            self.board = initial_board()
            self.turn = WHITE
            self.halfmove_clock = 0
        else:
            self._set_fen(fen)
        self.move_stack = []
        self._undo_stack = []
        self.history_hashes = [zobrist_hash(self.board, self.turn)]

    def _set_fen(self, fen):
        rows, turn, clock = fen.split(" ")
        self.board = [None] * NUM_SQUARES
        for row_idx, row in enumerate(rows.split("/")):
            rank = BOARD_SIZE - 1 - row_idx
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    color = WHITE if ch.isupper() else BLACK
                    self.board[square(file, rank)] = (
                        color,
                        SYMBOL_TO_PIECE[ch.upper()],
                    )
                    file += 1
        self.turn = WHITE if turn == "w" else BLACK
        self.halfmove_clock = int(clock)

    def fen(self):
        rows = []
        for rank in range(BOARD_SIZE - 1, -1, -1):
            row, empty = "", 0
            for file in range(BOARD_SIZE):
                piece = self.board[square(file, rank)]
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                letter = PIECE_SYMBOLS[piece[1]]
                row += letter if piece[0] == WHITE else letter.lower()
            if empty:
                row += str(empty)
            rows.append(row)
        turn = "w" if self.turn == WHITE else "b"
        return f"{'/'.join(rows)} {turn} {self.halfmove_clock}"

    def copy(self):
        clone = Board.__new__(Board)
        clone.board = self.board[:]
        clone.turn = self.turn
        clone.halfmove_clock = self.halfmove_clock
        clone.move_stack = self.move_stack[:]
        clone._undo_stack = self._undo_stack[:]
        clone.history_hashes = self.history_hashes[:]
        return clone

    @property
    def legal_moves(self):
        return legal_moves(self.board, self.turn)

    def piece_at(self, sq):
        return self.board[sq]

    def attacks(self, sq):
        piece = self.board[sq]
        if piece is None:
            return 0
        return sum(1 << t for t in pseudo_targets(self.board, sq))

    def push(self, move):
        undo = make_move(self.board, move)
        captured, moved = undo
        no_progress = (
            0 if (captured is not None or moved[1] == PAWN) else self.halfmove_clock + 1
        )
        self._undo_stack.append((undo, self.turn, self.halfmove_clock))
        self.turn = opponent(self.turn)
        self.halfmove_clock = no_progress
        self.move_stack.append(move)
        self.history_hashes.append(zobrist_hash(self.board, self.turn))

    def push_uci(self, text):
        move = Move.from_uci(text)
        for candidate in self.legal_moves:
            if (
                candidate.from_square == move.from_square
                and candidate.to_square == move.to_square
            ):
                if move.promotion is None or candidate.promotion == move.promotion:
                    self.push(candidate)
                    return candidate
        raise ValueError(f"illegal move: {text}")

    def pop(self):
        move = self.move_stack.pop()
        undo, prev_turn, prev_clock = self._undo_stack.pop()
        unmake_move(self.board, move, undo)
        self.turn = prev_turn
        self.halfmove_clock = prev_clock
        self.history_hashes.pop()
        return move

    def ply(self):
        return len(self.move_stack)

    def is_check(self):
        return in_check(self.board, self.turn)

    def king(self, color):
        sq = king_square(self.board, color)
        return None if sq < 0 else sq

    def is_repetition(self, count):
        current = self.history_hashes[-1]
        return self.history_hashes.count(current) >= count

    def outcome(self, claim_draw=True):
        return position_status(
            self.board, self.turn, self.history_hashes[:-1], self.halfmove_clock
        )

    def is_game_over(self, claim_draw=True):
        return self.outcome(claim_draw=claim_draw) is not None

    def san(self, move):
        piece = self.board[move.from_square]
        capture = self.board[move.to_square] is not None
        letter = "" if piece[1] == PAWN else PIECE_SYMBOLS[piece[1]]
        promo = f"={PIECE_SYMBOLS[move.promotion]}" if move.promotion else ""
        sep = "x" if capture else "-"
        return f"{letter}{square_name(move.from_square)}{sep}{square_name(move.to_square)}{promo}"
