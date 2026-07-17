import random

from dataset import dataset_to_samples


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


class DatasetBuffer:
    def __init__(self, dataset):
        self.dataset = dataset

    def sample_items(self, n):
        return dataset_to_samples(
            self.dataset.select(
                random.sample(range(len(self.dataset)), min(n, len(self.dataset)))
            )
        )

    def __len__(self):
        return len(self.dataset)


class DualRingBuffer:
    def __init__(self, pretrain_capacity=500000, rl_capacity=100000):
        self.pretrain_buf: RingBuffer | DatasetBuffer = RingBuffer(pretrain_capacity)
        self.rl_buf = RingBuffer(rl_capacity)

    def extend_pretrain(self, dataset):
        self.pretrain_buf = DatasetBuffer(dataset)

    def extend_rl(self, items):
        self.rl_buf.extend(items)

    def reset_rl(self):
        self.rl_buf.reset()

    def sample_pretrain(self, batch_size):
        return self.pretrain_buf.sample_items(batch_size)

    def sample_rl(self, batch_size):
        return self.rl_buf.sample_items(batch_size)
