import argparse
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import berserk
import chess

from config import Config, default_checkpoint_path, get_device
from model import load_checkpoint
from tree_search import mcts_move_with_visits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
for _noisy in ("urllib3", "requests", "berserk"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@dataclass(frozen=True)
class TimeControl:
    limit: int
    increment: int

    @property
    def perf(self) -> str:
        estimate = self.limit + 40 * self.increment
        if estimate < 29:
            return "ultraBullet"
        if estimate < 179:
            return "bullet"
        if estimate < 479:
            return "blitz"
        if estimate < 1499:
            return "rapid"
        return "classical"

    def __str__(self) -> str:
        return f"{self.perf} {self.limit}+{self.increment}"


PRESETS = {
    "bullet": TimeControl(60, 0),
    "blitz": TimeControl(180, 2),
    "rapid": TimeControl(600, 5),
    "classical": TimeControl(1800, 20),
}


def parse_time_control(spec: str) -> TimeControl:
    spec = spec.strip().lower()
    if spec in PRESETS:
        return PRESETS[spec]
    if "+" in spec:
        limit, _, increment = spec.partition("+")
        try:
            return TimeControl(int(limit), int(increment))
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"unknown time control {spec!r}; use a preset "
        f"({', '.join(PRESETS)}) or 'limit+increment' in seconds, e.g. 600+5"
    )


def clock_seconds(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


class Stopper:
    def __init__(self):
        self.requested = threading.Event()
        self.hard = threading.Event()

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError, AttributeError):
                logger.debug(f"could not install handler for signal {sig}")

    def _handle(self, signum, frame):  # noqa: ARG002
        if self.requested.is_set():
            logger.warning("Second interrupt — exiting immediately.")
            self.hard.set()
            raise KeyboardInterrupt
        self.requested.set()
        logger.warning(
            "Stop requested — finishing the current game, then exiting. "
            "Press Ctrl-C again to quit right away."
        )


def _stream_into_queue(make_stream, q: queue.Queue, stop: Stopper, label: str):
    attempt = 0
    while not stop.hard.is_set():
        try:
            for event in make_stream():
                q.put(event)
                if stop.hard.is_set():
                    break
        except Exception as e:
            if stop.hard.is_set():
                break
            delay = min(2 ** min(attempt, 3), 15)
            attempt += 1
            reason = type(e).__name__
            logger.info(f"{label} stream reconnecting in {delay}s ({reason})")
            time.sleep(delay)
            continue
        break
    q.put(None)


class ChallengeBot:
    def __init__(self, token: str, model, device, config, args, stopper: Stopper):
        self.token = token
        self.client = self.new_client()
        self.model = model
        self.device = device
        self.config = config
        self.args = args
        self.stopper = stopper
        self.nps: Optional[float] = None

        try:
            self.my_id = self.client.account.get()["id"]
        except berserk.exceptions.ResponseError as e:
            logger.critical(f"Authentication failed! Check your Lichess token: {e}")
            sys.exit(1)
        logger.info(f"Authenticated as @{self.my_id}")

        self.events: queue.Queue = queue.Queue()
        threading.Thread(
            target=_stream_into_queue,
            args=(
                self.new_client().bots.stream_incoming_events,
                self.events,
                self.stopper,
                "event",
            ),
            daemon=True,
        ).start()

    def new_client(self) -> berserk.Client:
        return berserk.Client(berserk.TokenSession(self.token))

    def ratings(self) -> dict:
        try:
            perfs = self.client.account.get().get("perfs", {})
        except Exception as e:
            logger.warning(f"Could not fetch ratings: {e}")
            return {}
        return {
            name: perf.get("rating")
            for name, perf in perfs.items()
            if isinstance(perf, dict) and perf.get("games")
        }

    def log_ratings(self, perfs=None):
        ratings = self.ratings()
        if not ratings:
            return
        shown = {k: v for k, v in ratings.items() if perfs is None or k in perfs}
        if shown:
            summary = ", ".join(f"{n} {r}" for n, r in sorted(shown.items()))
            logger.info(f"Ratings: {summary}")

    def candidate_bots(self, limit: int) -> list[dict]:
        try:
            bots = list(self.client.bots.get_online_bots(limit))
        except Exception as e:
            logger.error(f"Failed to list online bots: {e}")
            return []
        return [b for b in bots if b.get("id") != self.my_id]

    def opponents_for(self, bots: list[dict], tc: TimeControl) -> list[str]:
        rated = [
            b for b in bots if (b.get("perfs") or {}).get(tc.perf, {}).get("games")
        ]
        pool = rated or bots
        return [b["username"] for b in pool if b.get("username")]

    def run(self, time_controls: list[TimeControl], usernames: Optional[list[str]]):
        self.join_ongoing_games()

        discovered = None if usernames else self.candidate_bots(self.args.bot_limit)
        self.log_ratings()

        for round_index in range(1, self.args.rounds + 1):
            for tc in time_controls:
                if self.stopper.requested.is_set():
                    break
                pool = usernames or self.opponents_for(discovered or [], tc)
                if not pool:
                    logger.warning(f"No opponents available for {tc}; skipping.")
                    continue
                self.play_round(round_index, tc, pool)
            if self.stopper.requested.is_set():
                break

        self.log_ratings({tc.perf for tc in time_controls})
        if self.stopper.requested.is_set():
            logger.info("Stopped on request.")
        else:
            logger.info("Finished all rounds.")

    def play_round(self, round_index: int, tc: TimeControl, usernames: list[str]):
        started = time.monotonic()
        played = 0
        logger.info(
            f"=== Round {round_index} [{tc}] — up to {self.args.games_per_control} "
            f"game(s) from {len(usernames)} candidate(s) ==="
        )
        for index, username in enumerate(usernames, 1):
            if self.stopper.requested.is_set():
                break
            if played >= self.args.games_per_control:
                break
            elapsed = time.monotonic() - started
            if self.args.round_time_limit and elapsed >= self.args.round_time_limit:
                logger.info(
                    f"Round budget of {self.args.round_time_limit:.0f}s reached "
                    f"after {played} game(s); moving on."
                )
                break
            logger.info(
                f"[{tc.perf} round {round_index}] game {played + 1}/"
                f"{self.args.games_per_control}, candidate {index}/{len(usernames)}"
            )
            if self.challenge_and_play(username, tc):
                played += 1

        logger.info(
            f"=== Round {round_index} [{tc}] done: {played} game(s) in "
            f"{time.monotonic() - started:.0f}s ==="
        )
        self.log_ratings({tc.perf})

    def join_ongoing_games(self):
        try:
            ongoing = list(self.client.games.get_ongoing())
        except Exception as e:
            logger.error(f"Failed to check ongoing games: {e}")
            return
        for game in ongoing:
            if self.stopper.hard.is_set():
                return
            game_id = game.get("gameId") or game.get("id")
            if not game_id:
                continue
            logger.info(f"Resuming ongoing game [{game_id}] before challenging.")
            self.play_game(game_id)

    def challenge_and_play(self, username: str, tc: TimeControl) -> bool:
        logger.info(f"Challenging @{username} to {tc}...")
        try:
            challenge = self.client.challenges.create(
                username=username,
                rated=not self.args.casual,
                clock_limit=tc.limit,
                clock_increment=tc.increment,
                color="random",
                variant="standard",
            )
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Could not challenge @{username}: {e}")
            return False

        challenge_id = (challenge.get("challenge") or challenge).get("id")
        game_id = self.await_challenge(challenge_id, username)
        if game_id is None:
            return False
        self.play_game(game_id)
        return True

    def await_challenge(self, challenge_id, username: str) -> Optional[str]:
        deadline = time.monotonic() + self.args.challenge_timeout
        withdrawn = False
        while True:
            if self.stopper.hard.is_set():
                return None

            if not withdrawn and self.stopper.requested.is_set():
                logger.info("Stop requested; withdrawing the pending challenge.")
                withdrawn = True
                self.cancel_challenge(challenge_id)
                deadline = min(deadline, time.monotonic() + self.args.cancel_grace)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if withdrawn:
                    return None
                logger.info(
                    f"@{username} did not respond within "
                    f"{self.args.challenge_timeout:.0f}s; cancelling."
                )
                withdrawn = True
                self.cancel_challenge(challenge_id)
                deadline = time.monotonic() + self.args.cancel_grace
                continue

            try:
                event = self.events.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if event is None:
                logger.error("Event stream closed.")
                if not withdrawn:
                    self.cancel_challenge(challenge_id)
                return None

            event_type = event.get("type")
            if event_type == "gameStart":
                game = event.get("game", {})
                game_id = game.get("gameId") or game.get("id")
                if game_id:
                    if withdrawn:
                        logger.info(
                            f"@{username} accepted before the cancel landed; "
                            f"playing [{game_id}] out."
                        )
                    return game_id
            elif event_type in ("challengeDeclined", "challengeCanceled"):
                declined = (event.get("challenge") or {}).get("id")
                if declined == challenge_id or declined is None:
                    logger.info(f"@{username} declined the challenge.")
                    return None
            elif event_type == "challenge":
                self.decline_incoming(event.get("challenge") or {}, challenge_id)

    def cancel_challenge(self, challenge_id):
        if not challenge_id:
            return
        try:
            self.client.challenges.cancel(challenge_id)
        except berserk.exceptions.ResponseError as e:
            logger.debug(f"Could not cancel challenge {challenge_id}: {e}")

    def decline_incoming(self, challenge: dict, pending_id=None):
        challenge_id = challenge.get("id")
        if not challenge_id:
            return
        challenger = (challenge.get("challenger") or {}).get("id")
        if (
            challenge_id == pending_id
            or challenger == self.my_id
            or challenge.get("direction") == "out"
        ):
            return
        logger.info(f"Declining incoming challenge {challenge_id} (busy challenging).")
        try:
            self.client.challenges.decline(challenge_id)
        except berserk.exceptions.ResponseError as e:
            logger.debug(f"Could not decline {challenge_id}: {e}")

    def play_game(self, game_id: str):
        logger.info(f"Playing game [{game_id}]...")
        states: queue.Queue = queue.Queue()
        stream_client = self.new_client()
        threading.Thread(
            target=_stream_into_queue,
            args=(
                lambda: stream_client.bots.stream_game_state(game_id),
                states,
                self.stopper,
                f"game {game_id}",
            ),
            daemon=True,
        ).start()

        board = chess.Board()
        bot_color: Optional[chess.Color] = None

        while True:
            if self.stopper.hard.is_set():
                logger.warning(f"Abandoning game [{game_id}] on hard stop.")
                return
            try:
                event = states.get(timeout=1.0)
            except queue.Empty:
                continue
            if event is None:
                logger.info(f"Game [{game_id}] stream ended.")
                return

            event_type = event.get("type")
            if event_type == "gameFull":
                initial = event.get("initialFen", "startpos")
                board = chess.Board() if initial == "startpos" else chess.Board(initial)
                bot_color = (
                    chess.WHITE
                    if (event.get("white") or {}).get("id") == self.my_id
                    else chess.BLACK
                )
                logger.info(
                    f"[{game_id}] playing as "
                    f"{'white' if bot_color == chess.WHITE else 'black'}"
                )
                state = event.get("state", {})
            elif event_type == "gameState":
                state = event
            elif event_type == "chatLine":
                continue
            else:
                continue

            self.apply_moves(board, state.get("moves", ""))

            status = state.get("status", "started")
            if status != "started":
                logger.info(
                    f"[{game_id}] finished: {status}"
                    + (f" ({state['winner']} wins)" if state.get("winner") else "")
                )
                return

            if bot_color is not None and board.turn == bot_color:
                if not self.play_move(game_id, board, bot_color, state):
                    return

    def apply_moves(self, board: chess.Board, moves_str: str):
        moves = moves_str.split()
        if len(moves) < board.ply():
            board.reset()
        for move_uci in moves[board.ply() :]:
            try:
                board.push_uci(move_uci)
            except ValueError:
                logger.error(f"Illegal move in stream: {move_uci}")
                return

    def play_move(self, game_id, board, bot_color, state) -> bool:
        if board.is_game_over():
            return True

        budget = self.time_budget(board, bot_color, state)
        started = time.monotonic()
        search_budget = budget * self.args.time_safety
        try:
            move, visits = mcts_move_with_visits(
                board,
                self.model,
                self.device,
                self.config,
                num_simulations=self.simulation_cap(search_budget),
                deadline=started + search_budget,
            )
        except Exception as e:
            logger.error(f"[{game_id}] search failed: {e}", exc_info=True)
            return False

        elapsed = max(time.monotonic() - started, 1e-6)
        if visits > 0:
            observed = visits / elapsed
            self.nps = observed if self.nps is None else 0.7 * self.nps + 0.3 * observed
        logger.info(
            f"[{game_id}] {move.uci()} — {visits} sims in {elapsed:.2f}s "
            f"(budget {budget:.2f}s)"
        )

        try:
            self.client.bots.make_move(game_id, move.uci())
        except berserk.exceptions.ResponseError as e:
            logger.error(f"[{game_id}] Lichess rejected {move.uci()}: {e}")
            return False
        return True

    def time_budget(self, board, bot_color, state) -> float:
        remaining = clock_seconds(state.get("wtime" if bot_color else "btime"))
        increment = clock_seconds(state.get("winc" if bot_color else "binc")) or 0.0
        if remaining is None:
            return self.args.default_move_time

        horizon = max(self.args.moves_floor, 45.0 - board.ply() / 2.0)
        reserve = min(self.args.time_reserve, remaining * 0.25)
        usable = max(remaining - reserve, 0.0)
        budget = self.args.time_fraction * (usable / horizon + 0.5 * increment)
        budget = min(budget, usable * 0.5, self.args.max_move_time)
        return max(budget - self.args.move_overhead, self.args.min_move_time)

    def simulation_cap(self, budget: float) -> int:
        if self.nps is None:
            return self.config.inference_mcts_simulations
        estimate = int(1.25 * budget * self.nps)
        return max(self.args.min_simulations, min(self.args.max_simulations, estimate))


def main():
    parser = argparse.ArgumentParser(description="Lichess challenge bot")
    parser.add_argument("token", help="Lichess API OAuth token")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=default_checkpoint_path(),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--stockfish-path",
        default="/usr/games/stockfish",
        help="Path to stockfish binary",
    )
    parser.add_argument(
        "--opponents",
        default=None,
        help="Comma-separated usernames to challenge, in order. "
        "If omitted, online bots are discovered automatically.",
    )
    parser.add_argument(
        "--time-controls",
        default="bullet,blitz,rapid",
        help="Comma-separated time controls to play, in order. Each is a preset "
        f"({', '.join(PRESETS)}) or 'limit+increment' in seconds, e.g. 600+5.",
    )
    parser.add_argument("--rounds", type=int, default=1, help="Passes over the list")
    parser.add_argument(
        "--games-per-control",
        type=int,
        default=4,
        help="Maximum games to play per time control per round",
    )
    parser.add_argument(
        "--round-time-limit",
        type=float,
        default=0.0,
        help="Soft wall-clock budget in seconds per time control per round "
        "(0 disables; a game in progress is always finished)",
    )
    parser.add_argument(
        "--challenge-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for a challenge to be accepted before cancelling it",
    )
    parser.add_argument(
        "--cancel-grace",
        type=float,
        default=5.0,
        help="Seconds to keep watching after cancelling a challenge, in case the "
        "opponent accepted it just before the cancel landed",
    )
    parser.add_argument(
        "--bot-limit", type=int, default=200, help="Online bots to consider"
    )
    parser.add_argument(
        "--casual", action="store_true", help="Play unrated games (no rating changes)"
    )
    parser.add_argument("--min-move-time", type=float, default=0.05)
    parser.add_argument("--max-move-time", type=float, default=10.0)
    parser.add_argument(
        "--time-safety",
        type=float,
        default=0.9,
        help="Fraction of the move budget the search is allowed to use",
    )
    parser.add_argument(
        "--time-fraction",
        type=float,
        default=1.0,
        help="Scales every move budget; below 1.0 plays faster",
    )
    parser.add_argument(
        "--moves-floor",
        type=float,
        default=25.0,
        help="Never plan on having fewer than this many moves left to play",
    )
    parser.add_argument(
        "--time-reserve",
        type=float,
        default=3.0,
        help="Seconds of clock held back from the move budget",
    )
    parser.add_argument(
        "--move-overhead",
        type=float,
        default=0.3,
        help="Seconds reserved per move for network latency",
    )
    parser.add_argument(
        "--default-move-time",
        type=float,
        default=1.0,
        help="Move time used when the game stream reports no clock",
    )
    parser.add_argument("--min-simulations", type=int, default=32)
    parser.add_argument(
        "--max-simulations",
        type=int,
        default=200000,
        help="Hard node ceiling per move; normally the clock budget binds first",
    )
    args = parser.parse_args()

    time_controls = [parse_time_control(s) for s in args.time_controls.split(",") if s]
    if not time_controls:
        parser.error("--time-controls must name at least one time control")

    config = Config(stockfish_path=args.stockfish_path)
    device = get_device()
    try:
        model = load_checkpoint(args.checkpoint, device, config)
    except Exception as e:
        logger.critical(f"Failed to load checkpoint {args.checkpoint}: {e}")
        sys.exit(1)
    logger.info(f"Loaded checkpoint {args.checkpoint}")

    stopper = Stopper()
    stopper.install()
    bot = ChallengeBot(args.token, model, device, config, args, stopper)
    usernames = args.opponents.split(",") if args.opponents else None
    logger.info("Time controls: " + ", ".join(str(tc) for tc in time_controls))

    try:
        bot.run(time_controls, usernames)
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
