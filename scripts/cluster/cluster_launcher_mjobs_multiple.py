#!/usr/bin/env python3
"""
submit_experiments.py – fire off each experiment as an sbatch job.

Typical use:
    python submit_experiments.py --model css dur_small --partition gpu --gpus 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
import time
import shlex
from pathlib import Path
from datetime import datetime

from model_cards import MODELS
from experiments import EXPERIMENTS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

TRAIN = Path(__file__).resolve().parent.parent / "main.py"


def parse_args() -> argparse.Namespace:
    model_names = list(MODELS.keys())  # ← grab once

    p = argparse.ArgumentParser(
        description="Submit experiments to Slurm",
        epilog=f"Available models: {', '.join(model_names)}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --model accepts 1‒N names, still validated by choices
    p.add_argument(
        "--model",
        metavar="MODEL",
        nargs="+",
        choices=model_names,
        help="One or more model names (see list below)",
    )

    # Quick “show models then exit” helper ------------------------------
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print all model names and exit",
    )

    # ---- Slurm knobs ---------------------------------------------------
    p.add_argument("--partition", default="gpu")
    p.add_argument("--account", default=None)
    p.add_argument("--cpus", type=int, default=5)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--mem", default="3G")
    p.add_argument("--time", default="03:59:00")
    p.add_argument(
        "--output-dir",
        default="./cluster_runs",
        help="Root dir for logs & model checkpoints",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch scripts and save them, but do not submit",
    )

    args = p.parse_args()

    # Handle --list-models early and quit
    if args.list_models:
        print("\n".join(model_names))
        sys.exit(0)

    # With --list-models gone, --model is required
    if not args.model:
        p.error("--model is required unless --list-models is used")

    return args


# --------------------------------------------------------------------------- #
# Slurm helpers                                                               #
# --------------------------------------------------------------------------- #
def slurm_script(header: str, body: str) -> str:
    return textwrap.dedent(
        f"""\
#!/bin/bash
{header}

set -euo pipefail
module load stack/2024-05  gcc/13.2.0
module load ffmpeg
module load python/3.11.6_cuda
source /cluster/home/franre/set_up.sh

echo "--- Slurm Job Info ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "----------------------"
echo "Executing command: {body}"
echo "--- Python Script Output Starts ---"
{body}
echo "--- Python Script Output Ends ---"
echo "Job finished with exit code $?"
"""
    )


def build_header(
    idx: int, job_name_prefix: str, args: argparse.Namespace, log_dir: Path
) -> str:
    lines = [
        f"#SBATCH --job-name={job_name_prefix}_{idx}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --gpus={args.gpus}",
        f"#SBATCH --mem-per-cpu={args.mem}",
        f"#SBATCH --time={args.time}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    return "\n".join(lines)


def build_python_command(base_cfg: dict) -> str:
    parts = [sys.executable, str(TRAIN.resolve())]
    for k, v in base_cfg.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        elif v is not None:
            parts.extend([flag, str(v)])
    return " ".join(shlex.quote(p) for p in parts)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def submit_for_model(
    model_name: str, args: argparse.Namespace, batch_stamp: str
) -> None:
    """Generate & submit sbatch scripts for a single model."""
    # Resolve model type (just like before)
    model_type = "duration" if model_name.startswith("dur") else "saccade"

    experiment_group_name = f"{model_name}_{batch_stamp}"
    root = (
        Path(__file__).parent.parent
        / args.output_dir
        / model_type
        / experiment_group_name
    )

    slurm_log_dir = root / "slurm_logs"
    sbatch_scripts_dir = root / "sbatch_scripts"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    sbatch_scripts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model_name} ===")
    print(f"Results/logs → {root}")

    try:
        model_base_cfg = MODELS[model_name]
    except KeyError:
        sys.exit(
            f"Model '{model_name}' missing from MODELS mapping. "
            f"Available: {list(MODELS.keys())}"
        )

    # Each EXPERIMENTS entry is merged with the base model config
    experiment_configs = [
        {**model_base_cfg, **exp_cfg, "experiment_dir": str(root)}
        for exp_cfg in EXPERIMENTS
    ]

    for i, cfg in enumerate(experiment_configs, 1):
        body = build_python_command(cfg)
        job_name_prefix = cfg.get("job_name_prefix", model_name)
        header = build_header(i, job_name_prefix, args, slurm_log_dir)
        script_content = slurm_script(header, body)

        sbatch_path = sbatch_scripts_dir / f"{job_name_prefix}_{i}.sbatch"
        sbatch_path.write_text(script_content)
        sbatch_path.chmod(0o755)
        print(f"  • saved {sbatch_path}")

        if args.dry_run:
            print(f"    (dry-run) would submit: sbatch {sbatch_path}")
            continue

        print(f"    submitting {i}/{len(experiment_configs)} …")
        try:
            subprocess.run(["sbatch", str(sbatch_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"    ERROR submitting {sbatch_path.name}: {e}")
        time.sleep(0.1)  # gentle spacing


def main() -> None:
    args = parse_args()
    if not TRAIN.exists():
        sys.exit(f"Training script not found: {TRAIN.resolve()}")

    batch_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]

    for model_name in args.model:  # <─ iterate over every requested model
        submit_for_model(model_name, args, batch_stamp)

    if args.dry_run:
        print("\n(dry-run) scripts are ready; nothing was submitted.")
    else:
        print(
            "\n✅ All requested models processed. Check `squeue -u $USER` for status."
        )


if __name__ == "__main__":
    main()
