import argparse
import logging
import sys
from typing import Optional

import berserk
import chess

from config import Config, default_checkpoint_path, get_device
from model import load_checkpoint
from policy import batched_policy_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def bot_move(board: chess.Board, model, device) -> chess.Move:
    logger.debug(f"Evaluating position: {board.fen()}")
    moves, _, _, _ = batched_policy_step([board], model, device, temperature=0.0)
    selected_move = moves[0]
    logger.info(f"Model selected move: {selected_move.uci()}")
    return selected_move


class LichessBot:
    def __init__(self, token: str, model, device):
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        self.model = model
        self.device = device

        try:
            account_info = self.client.account.get()
            self.my_id = account_info["id"]
            logger.info(f"Successfully authenticated as Lichess bot: @{self.my_id}")
        except berserk.exceptions.ResponseError as e:
            logger.critical(
                f"Authentication failed! Check your Lichess token. Error: {e}"
            )
            sys.exit(1)

    def challenge_opponent(self, target_username: str = "dala-1100"):
        logger.info(f"Sending standard rapid challenge to @{target_username}...")

        try:
            challenge = self.client.challenges.create(
                username=target_username,
                rated=False,
                clock_limit=600,
                clock_increment=0,
                color="random",
                variant="standard",
            )
            challenge_id = challenge.get("id")
            logger.info(f"Challenge created successfully! Challenge ID: {challenge_id}")
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Failed to create challenge against {target_username}: {e}")
            return

        logger.info("Checking for any immediately started or ongoing games...")
        if self.check_and_join_ongoing_game():
            return

        logger.info(
            "No active games found yet. Listening for incoming Lichess event stream..."
        )
        try:
            for event in self.client.bots.stream_incoming_events():
                logger.debug(f"Received event: {event}")

                event_type = event.get("type")
                if event_type == "challengeDeclined":
                    logger.warning("Challenge was declined by the opponent.")
                    break
                elif event_type == "gameStart":
                    game = event.get("game", {})
                    game_id = game.get("id") or game.get("gameId")
                    if game_id:
                        logger.info(
                            f"Game started via event stream! Initializing game loop for ID: {game_id}"
                        )
                        self.play_game(game_id)
                        break
        except Exception as e:
            logger.error(f"Error reading event stream: {e}", exc_info=True)

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
                    moves_str = event.get("state", {}).get("moves", str())
                    self._update_board(board, moves_str)
                    self.maybe_move(game_id, board, bot_color)

                elif event_type == "gameState":
                    moves_str = event.get("moves", str())
                    board = chess.Board()
                    self._update_board(board, moves_str)

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
                move = bot_move(board, self.model, self.device)
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
        "--opponent", default="dala-1100", help="Username of the bot to challenge"
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

    bot = LichessBot(args.token, model, device)
    bot.challenge_opponent(target_username=args.opponent)


if __name__ == "__main__":
    main()
