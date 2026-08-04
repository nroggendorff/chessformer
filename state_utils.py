import numpy as np
import torch


def to_numpy_state(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: to_numpy_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_numpy_state(v) for v in obj]
    return obj


def from_numpy_state(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: from_numpy_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_numpy_state(v) for v in obj]
    return obj


def load_state(module, state):
    if state is not None:
        module.load_state_dict(from_numpy_state(state))
