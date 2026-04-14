import random
from typing import Optional

import numpy as np
import torch


def seed_rngs(seed: Optional[int], diff_seed=0):
    """Seed random number generators.

    Args:
        seed: int | None
            Seed to use. None to not seed the RNGs (for compatibility).
        diff_seed: int
            Difference of seed in different RNGs. Default is 0 (same seed for all RNGs).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed + diff_seed)
        torch.manual_seed(seed + diff_seed * 2)
