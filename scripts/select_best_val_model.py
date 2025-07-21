from pathlib import Path
import pdb
import shutil
from typing import Union, List

import pandas as pd

__all__ = ["collect_best_models"]

# -----------------------------------------------------------------------------
# Internal helper – unchanged core logic, but *only* for a single experiment dir
# -----------------------------------------------------------------------------


def _collect_best_models_single_experiment(
    experiment_dir: Path, model_type: str = "saccade"
) -> pd.DataFrame:
    """Process **one** experiment directory (which must contain
    ``sbatch_scripts`` and ``slurm_logs``).

    Creates/updates ``<experiment>/best_model`` with:
      * ``best_metrics_summary.csv`` – one‑row‑per‑run summary
      * copied artefacts from the run with the smallest
        ``Val_Loss_{model_type.upper()}_Mean``.

    Returns the summary *DataFrame* so that callers can still aggregate if they
    wish, but **does not write any aggregate file**.
    """

    best_rows = []
    best_dir = None
    best_val = float("inf")
    metric_col = f"Val_Loss_{model_type.upper()}_Mean"

    for subdir in experiment_dir.iterdir():
        # Skip auxiliary folders common to every experiment
        if subdir.name in {"sbatch_scripts", "slurm_logs"}:
            continue
        if not subdir.is_dir():
            continue  # ignore stray files

        metrics_path = subdir / "metrics.csv"
        if not metrics_path.is_file():
            print(f"[collect_best_models] {metrics_path} not found – skipped")
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception as exc:
            print(f"[collect_best_models] Failed to read {metrics_path}: {exc}")
            continue

        if metric_col not in df.columns:
            print(
                f"[collect_best_models] Column '{metric_col}' missing in {metrics_path}"
            )
            continue

        # Row with the minimum validation loss for *this* run
        min_idx = df[metric_col].idxmin()
        best_row = df.loc[min_idx].copy()
        best_row["Run_Directory"] = subdir.name
        best_rows.append(best_row)

        # Track the best run globally for artefact copying
        current_val = best_row[metric_col]
        if current_val < best_val:
            best_val = current_val
            best_dir = subdir

    # --------------------------------------------------------------
    # Aggregate & save summary inside <experiment_dir>/best_model
    # --------------------------------------------------------------
    if not best_rows:
        raise RuntimeError(f"No valid metrics.csv files found in {experiment_dir}")

    summary_df = pd.DataFrame(best_rows).sort_values(by=[metric_col])

    best_model_dir = experiment_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = best_model_dir / "best_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[collect_best_models] Summary saved to {summary_csv}")

    if best_dir is None:
        raise RuntimeError("Could not determine the best directory (no rows?)")

    # Copy artefacts ------------------------------------------------
    src_model_dir = best_dir / f"best_{model_type}_model"
    dst_model_dir = best_model_dir / f"best_{model_type}_model"
    if src_model_dir.is_dir():
        shutil.copytree(src_model_dir, dst_model_dir, dirs_exist_ok=True)
        print(
            f"[collect_best_models] Copied directory {src_model_dir} → {dst_model_dir}"
        )
    else:
        print(f"[collect_best_models] {src_model_dir} does not exist – skipped")

    mp4_files = list(best_dir.glob("*.mp4"))
    for mp4 in mp4_files:
        shutil.copy2(mp4, best_model_dir / mp4.name)
        print(f"[collect_best_models] Copied file {mp4.name} to best_model dir")
    if not mp4_files:
        print("[collect_best_models] No *.mp4 files found – skipped")

    train_log = best_dir / "train.log"
    if train_log.is_file():
        shutil.copy2(train_log, best_model_dir / train_log.name)
        print("[collect_best_models] Copied train.log")
    else:
        print("[collect_best_models] train.log not found – skipped")

    plot_train = best_dir / "plot_train.png"
    if plot_train.is_file():
        shutil.copy2(plot_train, best_model_dir / plot_train.name)
        print("[collect_best_models] Copied plot_train.png")
    else:
        print("[collect_best_models] plot_train.png not found – skipped")

    plot_val = best_dir / "plot_val.png"
    if plot_val.is_file():
        shutil.copy2(plot_val, best_model_dir / plot_val.name)
        print("[collect_best_models] Copied plot_val.png")
    else:
        print("[collect_best_models] plot_val.png not found – skipped")

    return summary_df


# -----------------------------------------------------------------------------
# Public API – now detects whether *root_dir* is a single experiment or a parent
# -----------------------------------------------------------------------------


def collect_best_models(
    root_dir: Union[str, Path], model_type: str = "saccade"
) -> List[pd.DataFrame]:
    """Run the *best‑model finder* in every qualifying child directory.

    A *qualifying* directory is one that contains **both** ``sbatch_scripts`` and
    ``slurm_logs``.  For each such directory we invoke the same logic that was
    previously run when you pointed the script at one experiment.

    Parameters
    ----------
    root_dir : str | Path
        Parent directory containing multiple experiment folders **or** one
        experiment folder itself.
    model_type : {"saccade", "duration"}
        Which metric column to look at.

    Returns
    -------
    list[pandas.DataFrame]
        A list of per‑experiment summary DataFrames (one per processed
        experiment).  Nothing is written globally – each experiment writes its
        own ``best_model/best_metrics_summary.csv`` only.
    """

    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a valid directory")

    sentinel_dirs = {"sbatch_scripts", "slurm_logs"}

    # ------------------------------------------------------------------
    # If *root* itself is an experiment directory, just run the helper
    # ------------------------------------------------------------------
    if all((root / d).is_dir() for d in sentinel_dirs):
        df = _collect_best_models_single_experiment(root, model_type)
        return [df]

    # ------------------------------------------------------------------
    # Otherwise, treat *root* as the parent of many experiments
    # ------------------------------------------------------------------
    experiment_dirs: List[Path] = [
        d
        for d in root.iterdir()
        if d.is_dir() and all((d / s).is_dir() for s in sentinel_dirs)
    ]

    if not experiment_dirs:
        raise RuntimeError(
            f"No experiment directories containing {sentinel_dirs} found under {root}"
        )

    summaries = []
    for exp_dir in experiment_dirs:
        print(f"[collect_best_models] Processing experiment: {exp_dir}")
        df = _collect_best_models_single_experiment(exp_dir, model_type)
        summaries.append(df)

    # Nothing extra is saved here – each experiment already wrote its own CSV
    return summaries


# -----------------------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect the best models inside each experiment directory "
        "that lives under a parent folder. A qualifying experiment must contain "
        "both 'sbatch_scripts' and 'slurm_logs'."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to a parent directory *or* to a single experiment directory.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--saccade",
        action="store_true",
        help="Process saccade models (default column Val_Loss_SACCADE_Mean).",
    )
    group.add_argument(
        "--duration",
        action="store_true",
        help="Process duration models (default column Val_Loss_DURATION_Mean).",
    )

    args = parser.parse_args()
    model_type = "saccade" if args.saccade else "duration"

    collect_best_models(args.root_dir, model_type)
    print("Done.")
