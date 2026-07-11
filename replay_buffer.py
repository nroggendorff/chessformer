import random

from dataset import dataset_to_samples


class RingBuffer:
    def __init__(self, capacity):
        self.capacity, self.buf, self.pos = capacity, [], 0

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

    def sample(self, batch_size, mix_ratio=0.5):
        if mix_ratio <= 0.0:
            return self.rl_buf.sample_items(batch_size)
        if not len(self.rl_buf):
            return self.pretrain_buf.sample_items(batch_size)
        if not len(self.pretrain_buf):
            return self.rl_buf.sample_items(batch_size)

        p_size = min(int(batch_size * mix_ratio), len(self.pretrain_buf))
        r_size = min(batch_size - p_size, len(self.rl_buf))
        batch = self.pretrain_buf.sample_items(p_size) + self.rl_buf.sample_items(
            r_size
        )
        random.shuffle(batch)
        return batch
