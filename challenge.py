import argparse
import logging
import sys
from typing import Optional

import berserk
import chess
from tqdm import tqdm

from config import Config, default_checkpoint_path, get_device
from model import load_checkpoint
from tree_search import mcts_move

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def bot_move(board: chess.Board, model, device, config) -> chess.Move:
    logger.debug(f"Evaluating position: {board.fen()}")
    selected_move = mcts_move(board, model, device, config)
    logger.info(f"Model selected move: {selected_move.uci()}")
    return selected_move


class LichessBot:
    def __init__(self, token: str, model, device, config):
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        self.model = model
        self.device = device
        self.config = config

        try:
            self.my_id = self.client.account.get()["id"]
            logger.info(f"Successfully authenticated as Lichess bot: @{self.my_id}")
        except berserk.exceptions.ResponseError as e:
            logger.critical(
                f"Authentication failed! Check your Lichess token. Error: {e}"
            )
            sys.exit(1)

    def list_candidate_bots(self, limit: int = 200) -> list[str]:
        return [
            b["username"]
            for b in self.client.bots.get_online_bots(limit)
            if b.get("id") != self.my_id and not b.get("playing", False)
        ]

    def challenge_bots(self, usernames: list[str]):
        if self.check_and_join_ongoing_game():
            return

        stream = self.client.bots.stream_incoming_events()
        for username in tqdm(usernames, desc="Challenging bots"):
            self._send_challenge(username) and self._await_challenge_outcome(stream)

        logger.info("Finished challenging the candidate bot list.")

    def _send_challenge(self, username: str) -> bool:
        logger.info(f"Sending challenge to @{username}...")
        try:
            challenge = self.client.challenges.create(
                username=username,
                rated=True,
                clock_limit=60,
                clock_increment=0,
                color="random",
                variant="standard",
            )
            logger.info(
                f"Challenge created successfully! Challenge ID: {challenge.get('id')}"
            )
            return True
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Failed to create challenge against {username}: {e}")
            return False

    def _await_challenge_outcome(self, stream) -> bool:
        for event in stream:
            logger.debug(f"Received event: {event}")
            event_type = event.get("type")

            if event_type == "challengeDeclined":
                logger.warning("Challenge was declined by the opponent.")
                return False

            if event_type == "gameStart":
                game = event.get("game", {})
                game_id = game.get("id") or game.get("gameId")
                if game_id:
                    logger.info(
                        f"Game started! Initializing game loop for ID: {game_id}"
                    )
                    self.play_game(game_id)
                    return True

        return False

    def check_and_join_ongoing_game(self) -> bool:
        try:
            ongoing_games = self.client.games.get_ongoing()
            if ongoing_games:
                game_id = ongoing_games[0].get("gameId")
                logger.info(
                    f"Found an active ongoing game [{game_id}]. Joining immediately!"
                )
                self.play_game(game_id)
                return True
        except Exception as e:
            logger.error(f"Failed to check ongoing games: {e}")
        return False

    def play_game(self, game_id: str):
        board = chess.Board()
        bot_color: Optional[chess.Color] = None

        logger.info(f"Connecting to game stream [{game_id}]...")
        try:
            for event in self.client.bots.stream_game_state(game_id):
                logger.debug(f"Game event: {event}")
                event_type = event.get("type")

                if event_type == "gameFull":
                    white_id = event.get("white", {}).get("id")
                    bot_color = chess.WHITE if white_id == self.my_id else chess.BLACK
                    color_str = "WHITE" if bot_color == chess.WHITE else "BLACK"
                    logger.info(
                        f"Game full state received. Bot is playing as {color_str}."
                    )

                    board = chess.Board()
                    self._update_board(
                        board, event.get("state", {}).get("moves", str())
                    )
                    self.maybe_move(game_id, board, bot_color)

                elif event_type == "gameState":
                    board = chess.Board()
                    self._update_board(board, event.get("moves", str()))

                    status = event.get("status")
                    if status and status != "started":
                        logger.info(f"Game over! Final status: {status}")
                        break

                    self.maybe_move(game_id, board, bot_color)

                elif event_type == "chatLine":
                    logger.info(f"[Chat] {event.get('username')}: {event.get('text')}")

        except Exception as e:
            logger.error(
                f"Exception during gameplay in game {game_id}: {e}", exc_info=True
            )

    def _update_board(self, board: chess.Board, moves_str: str):
        if not moves_str.strip():
            return
        for move_uci in moves_str.strip().split():
            try:
                board.push_uci(move_uci)
            except ValueError:
                logger.error(f"Received illegal move from stream: {move_uci}")

    def maybe_move(
        self, game_id: str, board: chess.Board, bot_color: Optional[chess.Color]
    ):
        if bot_color is None:
            return

        if board.turn == bot_color and not board.is_game_over():
            logger.info("It is the bot's turn. Calculating move...")
            try:
                move = bot_move(board, self.model, self.device, self.config)
                logger.info(f"Submitting move {move.uci()} to Lichess...")
                self.client.bots.make_move(game_id, move.uci())
            except berserk.exceptions.ResponseError as e:
                logger.error(f"API rejected move {move.uci()}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error making move: {e}", exc_info=True)
        elif board.is_game_over():
            logger.info(f"Game concluded. Result: {board.result()}")


def main():
    parser = argparse.ArgumentParser(description="Lichess AI Bot Client")
    parser.add_argument("token", help="Lichess API OAuth Token")
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
    args = parser.parse_args()

    logger.info("Initializing configuration and loading model...")
    config = Config(stockfish_path=args.stockfish_path)
    device = get_device()

    try:
        model = load_checkpoint(args.checkpoint, device, config)
        logger.info("Model checkpoint loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model checkpoint at {args.checkpoint}: {e}")
        sys.exit(1)

    bot = LichessBot(args.token, model, device, config)
    usernames = (
        args.opponents.split(",") if args.opponents else bot.list_candidate_bots()
    )
    logger.info(f"Candidate bots to challenge: {usernames}")
    bot.challenge_bots(usernames)


if __name__ == "__main__":
    main()
