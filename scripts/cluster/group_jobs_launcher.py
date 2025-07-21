#!/usr/bin/env python3
"""
submit_experiments.py – fire off each experiment as a grouped sbatch job.

Typical use:
    python submit_experiments.py --model css dur_small --partition gpu --gpus 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
import shlex
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from model_cards import MODELS
from experiments import EXPERIMENTS


TRAIN = Path(__file__).resolve().parent.parent / "main.py"


def parse_args() -> argparse.Namespace:
    model_names = list(MODELS.keys())

    p = argparse.ArgumentParser(
        description="Submit grouped experiments to Slurm",
        epilog=f"Available models: {', '.join(model_names)}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--model",
        metavar="MODEL",
        nargs="+",
        choices=model_names,
        help="One or more model names (see list below)",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print all model names and exit",
    )

    # Slurm options
    p.add_argument("--partition", default="gpu")
    p.add_argument("--account", default=None)
    p.add_argument("--ntasks", type=int, default=5, help="Number of tasks per job")
    p.add_argument("--cpu", type=int, default=5)
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
        dest="dry_run",
        action="store_true",
        help="Print sbatch scripts but do not submit",
    )

    args = p.parse_args()
    if args.list_models:
        print("\n".join(model_names))
        sys.exit(0)
    if not args.model:
        p.error("--model is required unless --list-models is used")
    return args


def slurm_script(header: str, body: str) -> str:
    return textwrap.dedent(
        f"""\
#!/bin/bash
{header}

set -euo pipefail
module load stack/2024-05 gcc/13.2.0
module load ffmpeg
module load python/3.11.6_cuda
source /cluster/home/franre/set_up.sh

echo "--- Slurm Job Info ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Hostname: $(hostname)"
echo "Working Dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "----------------------"

# Run all experiments in parallel within this job
{body}

echo "All experiments completed with exit code $?"
"""
    )


def build_header(job_name: str, args: argparse.Namespace, log_dir: Path) -> str:
    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --ntasks={args.ntasks}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --cpus-per-task={args.cpu}",
        f"#SBATCH --gpus={args.gpus}",
        f"#SBATCH --mem-per-cpu={args.mem}",
        f"#SBATCH --time={args.time}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    return "\n".join(lines)


def build_python_command(cfg: dict) -> str:
    parts = [sys.executable, str(TRAIN.resolve())]
    for k, v in cfg.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        elif v is not None:
            parts.extend([flag, str(v)])
    return " ".join(shlex.quote(p) for p in parts)


def submit_grouped_for_model(
    model_name: str, args: argparse.Namespace, stamp: str
) -> None:
    model_type = "duration" if model_name.startswith("dur") else "saccade"
    group_name = f"{model_name}_{stamp}"
    root = Path(__file__).parent.parent / args.output_dir / model_type / group_name
    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model_name} grouped job ===")
    print(f"Logs → {root}")

    try:
        base_cfg = MODELS[model_name]
    except KeyError:
        sys.exit(f"Model '{model_name}' not found; available: {list(MODELS.keys())}")

    configs = [{**base_cfg, **e, "experiment_dir": str(root)} for e in EXPERIMENTS]
    cmds = [build_python_command(c) for c in configs]

    # Build bash loop in batches of args.ntasks
    body_lines = [f"K={args.ntasks}", "i=0", "commands=("]
    for cmd in cmds:
        body_lines.append(f"    '{cmd}'")
    body_lines.append(")")
    body_lines.append('for c in "${commands[@]}"; do')
    body_lines.append('  echo "Running: $c"')
    body_lines.append("  $c &")
    body_lines.append("  ((i=(i+1)%K)) && wait")
    body_lines.append("done")
    body_lines.append("wait")
    body = "\n".join(body_lines)

    header = build_header(group_name, args, slurm_dir)
    script = slurm_script(header, body)

    sbatch_path = slurm_dir / f"{group_name}.sbatch"
    sbatch_path.write_text(script)
    sbatch_path.chmod(0o755)
    print(f"  • saved grouped script: {sbatch_path}")

    if args.dry_run:
        print(f"    (dry-run) sbatch {sbatch_path}")
    else:
        subprocess.run(["sbatch", str(sbatch_path)], check=True)


def main() -> None:
    args = parse_args()
    if not TRAIN.exists():
        sys.exit(f"Training script not found: {TRAIN.resolve()}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    for model in args.model:
        submit_grouped_for_model(model, args, stamp)

    if args.dry_run:
        print("\n(dry-run) nothing submitted.")
    else:
        print("\n✅ Grouped jobs submitted. Use `squeue -u $USER` to check.")


if __name__ == "__main__":
    main()
