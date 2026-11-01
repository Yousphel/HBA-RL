"""
Seed Management Utility
========================
Centralised seed control for full reproducibility across:
  - Python random
  - NumPy
  - PyTorch (CPU + GPU)
  - Gymnasium environment

Usage:
  - Every training run takes its seed from the command line (--seed) and prints it.
  - Reported results use 10 seeds (see the reproduction notebook / reproduce.sh).
"""

import random
import numpy as np
import torch
import os


def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.
    Call this ONCE at the very start of main() before
    anything else is initialised.

    Parameters
    ----------
    seed : int
        Random seed. Default = 42.
        For multi-run experiments use: 42, 123, 456
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # OS-level seed (affects some C extensions)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[Seed] Global seed set to {seed} "
          f"(torch={torch.initial_seed()}, "
          f"cuda={'yes' if torch.cuda.is_available() else 'no'})")


def get_env_seed(base_seed: int, episode: int) -> int:
    """
    Deterministic per-episode seed for TRAINING episodes.

    Revision 1: placed far above the evaluation seeds (42 .. 141) so that no training
    episode can ever coincide with an evaluation episode. The old formula
    base_seed + episode * 1000 gave training episode 0 the seeds 7, 42, 123 and 456,
    and 42 and 123 lie inside the evaluation range 42..141.
    Evaluation episodes use seed 42 + e for every method (common.EVAL_SEED_BASE).
    """
    return 10_000_000 + base_seed * 100_000 + episode
