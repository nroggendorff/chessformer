import datasets
import numpy as np

from data_generation import generate_pretrain_data

COLUMNS = (
    "board_input",
    "target_squares",
    "target_tokens",
    "target_weights",
    "value",
    "policy_weight",
    "value_weight",
)

DEFAULT_PATH = "pretrain_data"


def dataset_to_samples(ds):
    columns = ds.to_dict()
    return [
        (
            np.array(board_input, dtype=np.uint8),
            np.array(target_squares, dtype=np.uint8),
            np.array(target_tokens, dtype=np.uint8),
            np.array(target_weights, dtype=np.float32),
            value,
            policy_weight,
            value_weight,
        )
        for board_input, target_squares, target_tokens, target_weights, value, policy_weight, value_weight in zip(
            *(columns[name] for name in COLUMNS)
        )
    ]


def generate_pretrain_dataset(config, path=DEFAULT_PATH):
    ds = datasets.Dataset.from_generator(
        lambda: generate_pretrain_data(config), writer_batch_size=10000
    )
    ds.save_to_disk(path)
    print(f"Generated {len(ds)} pretraining positions, saved to {path}.")
    return ds


def load_pretrain_dataset(path=DEFAULT_PATH):
    return datasets.Dataset.load_from_disk(path)


if __name__ == "__main__":
    from config import Config

    generate_pretrain_dataset(Config())
