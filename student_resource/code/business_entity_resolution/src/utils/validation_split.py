"""S1-level train/validation split (Phase 8 dependency).

All candidate pairs for a given S1 entity must stay entirely in train or
entirely in validation -- never split by pair, to avoid leakage from a
pair-level random split (BUILDING.md Phase 8 / architecture report §12).
"""

from __future__ import annotations

import random


def split_s1_ids(
    s1_ids: list[str], val_fraction: float = 0.15, random_seed: int = 42
) -> tuple[set[str], set[str]]:
    """Deterministically split a list of S1 ids into (train_ids, val_ids)."""
    ids = sorted(set(s1_ids))
    rng = random.Random(random_seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids
