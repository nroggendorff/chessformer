import argparse
import threading

import berserk
import chess

from config import Config, default_checkpoint_path, get_device
from model import load_checkpoint
from tree_search import mcts_move


def bot_move(board, model, device, config):
    return mcts_move(board, model, device, config)


class LichessBot:
    def __init__(self, token, model, device, config):
        self.client = berserk.Client(berserk.TokenSession(token))
        self.model = model
        self.device = device
        self.config = config
        self.my_id = self.client.account.get()["id"]
        self.active_game_id = None
        self.lock = threading.Lock()

    def stream_events(self):
        print("listening for events", flush=True)
        for event in self.client.bots.stream_incoming_events():
            print(event, flush=True)
            if event["type"] == "challenge":
                self.handle_challenge(event["challenge"])
            elif event["type"] == "gameStart":
                self.handle_game_start(event["game"]["gameId"])

    def handle_challenge(self, challenge):
        challenge_id = challenge["id"]
        with self.lock:
            busy = self.active_game_id is not None
        if busy:
            print(
                f"declining {challenge_id}: already playing {self.active_game_id}",
                flush=True,
            )
            self.client.challenges.decline(challenge_id)
        elif challenge["variant"]["key"] != "standard":
            print(
                f"declining {challenge_id}: variant {challenge['variant']['key']}",
                flush=True,
            )
            self.client.challenges.decline(challenge_id)
        else:
            print(f"accepting {challenge_id}", flush=True)
            self.client.challenges.accept(challenge_id)

    def handle_game_start(self, game_id):
        with self.lock:
            if self.active_game_id is not None:
                return
            self.active_game_id = game_id
        print(f"game started: {game_id}", flush=True)
        threading.Thread(target=self.play_game, args=(game_id,), daemon=True).start()

    def play_game(self, game_id):
        try:
            for event in self.client.bots.stream_game_state(game_id):
                if event["type"] == "gameFull":
                    self.bot_color = (
                        chess.WHITE
                        if event["white"].get("id") == self.my_id
                        else chess.BLACK
                    )
                    self.board = (
                        chess.Board()
                        if event["initialFen"] == "startpos"
                        else chess.Board(event["initialFen"])
                    )
                    self.maybe_move(game_id, event["state"])
                elif event["type"] == "gameState":
                    self.maybe_move(game_id, event)
        except Exception as e:
            print(f"error in game {game_id}: {e}", flush=True)
        finally:
            with self.lock:
                self.active_game_id = None

    def maybe_move(self, game_id, state):
        for move in state["moves"].split()[self.board.ply() :]:
            self.board.push_uci(move)
        if (
            state.get("status", "started") == "started"
            and self.board.turn == self.bot_color
            and not self.board.is_game_over()
        ):
            move = bot_move(self.board, self.model, self.device, self.config)
            print(f"playing {move.uci()}", flush=True)
            self.client.bots.make_move(game_id, move.uci())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("token")
    parser.add_argument("checkpoint", nargs="?", default=default_checkpoint_path())
    parser.add_argument("--stockfish-path", default="/usr/games/stockfish")
    args = parser.parse_args()

    config = Config(stockfish_path=args.stockfish_path)
    device = get_device()
    model = load_checkpoint(args.checkpoint, device, config)

    LichessBot(args.token, model, device, config).stream_events()


if __name__ == "__main__":
    main()
