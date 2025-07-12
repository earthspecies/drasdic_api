"""
Misc. utils
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across torch, numpy, and random.

    Parameters
    ----------
    seed : int
        Seed value to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
