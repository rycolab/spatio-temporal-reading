from pathlib import Path
from typing import List, Tuple
import numpy as np


def load_test_results(
    root_dir: str | Path, model_type: str
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Walk through every *first-level* subdirectory in `root_dir`, find the
    `test_set_eval_*` folder, then read the two `.npy` files inside
    `loss_results_{model_type}_test`.

    Parameters
    ----------
    root_dir : str | pathlib.Path
        The directory whose immediate children will be scanned.
    model_type : str
        The model identifier (e.g. "cnn", "transformer"). Used to build the
        folder name and the two file names.

    Returns
    -------
    losses : list[np.ndarray]
        One entry per (sub-)experiment, in the same order they’re found.
    subset_flags : list[np.ndarray]
        The corresponding subset-flag arrays, same ordering as `losses`.

    Raises
    ------
    FileNotFoundError
        If **none** of the expected result files are found.
    """
    root_dir = Path(root_dir).expanduser().resolve()

    losses: dict[str, np.ndarray] = {}
    subset_flags: dict[str, np.ndarray] = {}

    for subdir in (d for d in root_dir.iterdir() if d.is_dir()):
        # look for `test_set_eval_*` directly inside this subdir
        for eval_dir in subdir.glob("test_set_eval*"):
            loss_dir = eval_dir / f"loss_results_{model_type}_test"
            if not loss_dir.is_dir():
                continue

            loss_file = loss_dir / f"test_loss_{model_type}.npy"
            flags_file = loss_dir / f"test_subsets_flags_{model_type}.npy"
            model_name = subdir.name

            if loss_file.exists() and flags_file.exists():
                losses[model_name] = np.load(loss_file)
                subset_flags[model_name] = np.load(flags_file)
            else:
                print(f"Skipping {loss_dir}: missing files.")

    if not losses:
        raise FileNotFoundError(
            f"No matching results for model_type='{model_type}' under {root_dir}"
        )

    return losses, subset_flags


def bootstrap_mean_difference(v1, v2, N, reduce=True):
    """
    Computes bootstrap resampling of the mean differences between v1 and v2.

    Parameters:
        v1 (array-like): First vector of negative likelihoods.
        v2 (array-like): Second vector of negative likelihoods.
        N (int): Number of bootstrap samples.

    Returns:
        v3 (np.ndarray): Bootstrap vector of mean differences of length N.
                         Each element is the mean of the difference (v1 - v2)
                         from one bootstrap sample.
    """
    # Convert inputs to NumPy arrays for indexing and computation
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Ensure the two vectors are of the same length
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("v1 and v2 must have the same length.")

    n = v1.shape[0]  # original sample size
    v3 = np.empty(N)  # initialize bootstrap results vector

    # Perform N bootstrap iterations
    for i in range(N):
        # Generate indices for bootstrap sample (with replacement)
        indices = np.random.choice(n, size=n, replace=True)
        # Compute the mean difference for the resampled data, preserving pairing
        if reduce == True:
            v3[i] = (v1[indices] - v2[indices]).mean()
        else:
            v3[i] = (v1[indices] - v2[indices]).sum()

    return v3
