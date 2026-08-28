import queue
import random
import threading

import numpy as np
import pyarrow as pa

from dataset import COLUMNS


class RingBuffer:
    def __init__(self, capacity):
        self.capacity, self.buf, self.pos = capacity, [], 0

    def reset(self):
        self.buf, self.pos = [], 0

    def extend(self, items):
        for item in items:
            if len(self.buf) < self.capacity:
                self.buf.append(item)
            else:
                self.buf[self.pos] = item
                self.pos = (self.pos + 1) % self.capacity

    def sample_items(self, n):
        return random.sample(self.buf, min(n, len(self.buf)))

    def __len__(self):
        return len(self.buf)


def _single_array(column):
    column = column.combine_chunks()
    return column.chunk(0) if isinstance(column, pa.ChunkedArray) else column


def _fixed_width(column, dtype):
    arr = _single_array(column)
    offsets = arr.offsets.to_numpy()
    values = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=dtype)
    return values[offsets[0] : offsets[-1]].reshape(len(arr), -1)


def _ragged(column, dtype):
    arr = _single_array(column)
    values = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=dtype)
    return values, arr.offsets.to_numpy()


def _ragged_pairs(column, dtype):
    outer = _single_array(column)
    inner = outer.values
    inner_offsets = inner.offsets.to_numpy()
    values = np.asarray(inner.values.to_numpy(zero_copy_only=False), dtype=dtype)
    return values, inner_offsets[outer.offsets.to_numpy()]


class _Chunk:
    __slots__ = (
        "boards",
        "legal",
        "legal_off",
        "pairs",
        "pairs_off",
        "probs",
        "probs_off",
        "values",
        "pw",
        "vw",
    )

    def __init__(self, table, start, count):
        sl = table.slice(start, count)
        self.boards = _fixed_width(sl.column(COLUMNS[0]), np.int64)
        self.legal, self.legal_off = _ragged_pairs(sl.column(COLUMNS[1]), np.uint8)
        self.pairs, self.pairs_off = _ragged_pairs(sl.column(COLUMNS[2]), np.uint8)
        self.probs, self.probs_off = _ragged(sl.column(COLUMNS[3]), np.float32)
        self.values = sl.column(COLUMNS[4]).to_numpy(zero_copy_only=False)
        self.pw = sl.column(COLUMNS[5]).to_numpy(zero_copy_only=False)
        self.vw = sl.column(COLUMNS[6]).to_numpy(zero_copy_only=False)

    def __len__(self):
        return len(self.values)

    def row(self, i):
        return (
            self.boards[i],
            self.legal[self.legal_off[i] : self.legal_off[i + 1]].reshape(-1, 2),
            self.pairs[self.pairs_off[i] : self.pairs_off[i + 1]].reshape(-1, 2),
            self.probs[self.probs_off[i] : self.probs_off[i + 1]],
            self.values[i],
            self.pw[i],
            self.vw[i],
        )


class DatasetBuffer:
    def __init__(self, dataset, pool_size=262144, chunk_size=16384, prefetch=2):
        self.dataset = dataset
        self.table = (
            dataset.data.table if hasattr(dataset.data, "table") else dataset.data
        )
        self.chunk_size = max(1, min(chunk_size, len(dataset)))
        self.pool_size = max(pool_size, 2 * self.chunk_size)
        self.pool = []
        self.served_since_admit = 0

        self._offsets = list(
            range(0, len(dataset) - self.chunk_size + 1, self.chunk_size)
        )
        random.shuffle(self._offsets)
        self._next_offset = 0
        self._queue = queue.Queue(maxsize=max(1, prefetch))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _read_chunk(self):
        if self._next_offset >= len(self._offsets):
            random.shuffle(self._offsets)
            self._next_offset = 0
        start = self._offsets[self._next_offset]
        self._next_offset += 1
        return _Chunk(self.table, start, self.chunk_size)

    def _produce(self):
        while not self._stop.is_set():
            chunk = self._read_chunk()
            while not self._stop.is_set():
                try:
                    self._queue.put(chunk, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _admit(self, block):
        try:
            chunk = self._queue.get(timeout=120) if block else self._queue.get_nowait()
        except queue.Empty:
            return False
        self.served_since_admit = 0
        rows = [(chunk, i) for i in range(len(chunk))]
        if len(self.pool) + len(rows) <= self.pool_size:
            self.pool.extend(rows)
        else:

            size = len(self.pool)
            for row in rows:
                self.pool[random.randrange(size)] = row
        return True

    def sample_items(self, n, require_policy=False):

        while len(self.pool) + self.chunk_size <= self.pool_size:
            self._admit(block=True)
        if self.served_since_admit >= self.chunk_size:
            self._admit(block=False)
        self.served_since_admit += n
        size = len(self.pool)
        out = []
        if require_policy:

            for _ in range(n * 32):
                if len(out) >= n:
                    break
                chunk, index = self.pool[random.randrange(size)]
                if chunk.pw[index] > 0:
                    out.append(chunk.row(index))
        while len(out) < n:
            chunk, index = self.pool[random.randrange(size)]
            out.append(chunk.row(index))
        return out

    def close(self):
        self._stop.set()

    def __len__(self):
        return len(self.dataset)


class DualRingBuffer:
    def __init__(self, pretrain_capacity=500000, rl_capacity=100000):
        self.pretrain_buf: RingBuffer | DatasetBuffer = RingBuffer(pretrain_capacity)
        self.rl_buf = RingBuffer(rl_capacity)

    def extend_pretrain(self, dataset, pool_size=262144, chunk_size=16384):
        self.pretrain_buf = DatasetBuffer(
            dataset, pool_size=pool_size, chunk_size=chunk_size
        )

    def extend_rl(self, items):
        self.rl_buf.extend(items)

    def reset_rl(self):
        self.rl_buf.reset()

    def sample_pretrain(self, batch_size, require_policy=False):
        if isinstance(self.pretrain_buf, DatasetBuffer):
            return self.pretrain_buf.sample_items(
                batch_size, require_policy=require_policy
            )
        return self.pretrain_buf.sample_items(batch_size)

    def sample_rl(self, batch_size):
        return self.rl_buf.sample_items(batch_size)
