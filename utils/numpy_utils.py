from typing import Optional

import numpy as np
from contextlib import contextmanager


@contextmanager
def numpy_random_seed(seed: Optional[int] = None):
    """
    Context manager to set the random state with a given seed (or None to make it non-deterministic).
    Restores the old state upon exiting.
    Args:
        seed: optional(int), the seed to set the random state.
    """
    # get the current random state
    old_state = np.random.get_state()

    # set the new random state
    np.random.seed(seed)

    try:
        yield
    finally:
        # restore the old state
        np.random.set_state(old_state)
