import pecan as pc


class Oracle:
    def __init__(self, depth=4, node_cap=None, time_limit=None):
        self.depth = depth
        self.node_cap = node_cap
        self.time_limit = time_limit
        self._engine = pc.Engine()

    def search(self, board, depth=None, node_cap=None, time_limit=None):
        return self._engine.search(
            board.board,
            board.turn,
            board.history_hashes,
            max_depth=depth or self.depth,
            node_cap=node_cap if node_cap is not None else self.node_cap,
            time_limit=time_limit if time_limit is not None else self.time_limit,
        )

    def play(self, board, depth=None, node_cap=None, time_limit=None):
        return self.search(board, depth, node_cap, time_limit)["move"]

    def score_cp(self, board, depth=None, node_cap=None):
        return self.search(board, depth, node_cap)["score"]
