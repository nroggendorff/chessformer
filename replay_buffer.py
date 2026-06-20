import random
from collections import deque


class DualRingBuffer:
    def __init__(self, pretrain_capacity=500000, rl_capacity=100000):
        self.pretrain_buf = deque(maxlen=pretrain_capacity)
        self.rl_buf = deque(maxlen=rl_capacity)

    def extend_pretrain(self, items):
        self.pretrain_buf.extend(items)

    def extend_rl(self, items):
        self.rl_buf.extend(items)

    def sample(self, batch_size, mix_ratio=0.5):
        if not self.rl_buf:
            return random.sample(self.pretrain_buf, batch_size)
        if not self.pretrain_buf:
            return random.sample(self.rl_buf, batch_size)

        p_size = min(int(batch_size * mix_ratio), len(self.pretrain_buf))
        r_size = min(batch_size - p_size, len(self.rl_buf))
        batch = random.sample(self.pretrain_buf, p_size) + random.sample(
            self.rl_buf, r_size
        )
        random.shuffle(batch)
        return batch
