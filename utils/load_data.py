"""Load image arrays from Hugging Face Hub."""

import numpy as np
from datasets import load_dataset

from config import HF_DATASET_REPO


def load_arrays(
    split: str = "train",
    repo_id: str = HF_DATASET_REPO,
    streaming: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image arrays and labels for the given split.

    Parameters
    ----------
    split : str
        One of "train", "test", "validation".
    repo_id : str
        Hugging Face dataset repository ID.
    streaming : bool
        If True, iterates without caching (for memory-constrained envs).
        If False (default), downloads full split to local HF cache.

    Returns
    -------
    X : np.ndarray, shape (N, 224, 224, 3), dtype uint8
    y : np.ndarray, shape (N,), dtype int64
    """
    if streaming:
        ds = load_dataset(repo_id, split=split, streaming=True)
        images, labels = [], []
        for example in ds:
            images.append(np.array(example["image"]))
            labels.append(example["label"])
        X = np.stack(images)
        y = np.array(labels)
    else:
        ds = load_dataset(repo_id, split=split)
        X = np.stack([np.array(ex["image"]) for ex in ds])
        y = np.array(ds["label"])

    return X, y
