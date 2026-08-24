import torch
from sbi.utils import BoxUniform
from . import config


def build_prior() -> BoxUniform:
    low = torch.tensor(
        [v["low"] for v in config.PARAMETER_SPACE.values()], dtype=torch.float32
    )
    high = torch.tensor(
        [v["high"] for v in config.PARAMETER_SPACE.values()], dtype=torch.float32
    )
    return BoxUniform(low=low, high=high)
