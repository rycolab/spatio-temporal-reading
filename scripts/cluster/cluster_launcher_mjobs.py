#!/usr/bin/env python3
"""
submit_experiments.py  –  fire off each experiment as an sbatch job.

Typical use:
    python submit_experiments.py --model css --partition gpu --gpus 1
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import textwrap
import time
import shlex
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Assume these are in a place accessible by the script, e.g., same directory or PYTHONPATH
from model_cards import MODELS
from experiments import EXPERIMENTS


TRAIN = Path(__file__).resolve().parent.parent.parent / "main.py"  # adjust if needed


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit experiments to Slurm")
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),  # Dynamically get choices from MODELS
    )
    p.add_argument(
        "--output-dir",
        default="./cluster_runs",  # Changed default to cluster_runs as per your output
        help="Root dir for logs & model checkpoints",
    )
    # ---- Slurm resource knobs (cluster-specific defaults) ----------------
    p.add_argument("--partition", default="gpu")
    p.add_argument("--account", default=None)  # None → omit line
    p.add_argument("--cpus", type=int, default=5)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--mem", default="3G")  # e.g. 4G, 16000M
    p.add_argument("--time", default="03:59:00")  # HH:MM:SS
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch scripts and save them, but do not submit",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def slurm_script(header: str, body: str) -> str:
    """Wrap SBATCH header + shell body in a complete bash script."""
    # The f"""\ is important. The actual content for dedent starts on the next line.
    # The shebang #!/bin/bash MUST be the first thing on its line in the template.
    return textwrap.dedent(
        f"""\
#!/bin/bash
{header}

set -euo pipefail
# Consider making module loads configurable or part of body if they vary
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
    """Return the SBATCH lines as a string (no leading #!)."""
    lines: list[str] = [
        f"#SBATCH --job-name={job_name_prefix}_{idx}",
        f"#SBATCH --output={log_dir}/%x_%j.out",  # %x job name, %j job ID
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
    """Builds the python command string with proper quoting for shell execution."""
    cmd_parts = [sys.executable, str(TRAIN.resolve())]  # Use resolved path for TRAIN

    for k, v in base_cfg.items():
        arg_name = f"--{k.replace('_', '-')}"

        if isinstance(v, bool):
            if v:
                cmd_parts.append(arg_name)
        elif v is None:
            pass
        else:
            cmd_parts.append(arg_name)
            cmd_parts.append(str(v))
    return " ".join(shlex.quote(str(p)) for p in cmd_parts)


def main() -> None:
    args = parse_args()
    if not TRAIN.exists():
        sys.exit(f"Training script not found: {TRAIN.resolve()}")

    if args.model.startswith("dur"):
        model_type = "duration"
    else:
        model_type = "saccade"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[
        :-3
    ]  # Include milliseconds
    experiment_group_name = f"{args.model}_{timestamp}"
    root = (
        Path(__file__).parent.parent.parent
        / args.output_dir
        / model_type
        / experiment_group_name
    )
    slurm_log_dir = root / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_scripts_dir = root / "sbatch_scripts"
    sbatch_scripts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results and logs will be stored in: {root}")
    print(f"Generated sbatch scripts will be saved in: {sbatch_scripts_dir}")
    print(f"Slurm stdout/stderr logs will be saved in: {slurm_log_dir}")

    try:
        model_base_cfg = MODELS[args.model]
    except KeyError:
        sys.exit(
            f"Model '{args.model}' missing from MODELS mapping. Available: {list(MODELS.keys())}"
        )

    experiment_configs = []
    for exp_specific_cfg in EXPERIMENTS:
        full_cfg = {**model_base_cfg, **exp_specific_cfg, "experiment_dir": str(root)}
        experiment_configs.append(full_cfg)

    if not experiment_configs:
        print(
            "No experiments found to submit for the selected model based on EXPERIMENTS definition."
        )
        return

    for i, cfg in enumerate(experiment_configs, 1):
        shell_cmd_body = build_python_command(cfg)
        job_name_prefix = cfg.get("job_name_prefix", args.model)
        job_actual_name = f"{job_name_prefix}_{i}"

        header = build_header(i, job_name_prefix, args, slurm_log_dir)
        script_content = slurm_script(header, shell_cmd_body)

        sbatch_script_path = sbatch_scripts_dir / f"{job_actual_name}.sbatch"

        try:
            with open(sbatch_script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            sbatch_script_path.chmod(0o755)
            print(f"Saved sbatch script to: {sbatch_script_path}")
        except IOError as e:
            print(f"ERROR: Could not write sbatch script to {sbatch_script_path}: {e}")
            continue

        if args.dry_run:
            print(
                f"DRY RUN: Would submit script for job {job_actual_name}. Script at: {sbatch_script_path}"
            )
            continue

        print(
            f"Submitting exp {i}/{len(experiment_configs)} ({job_actual_name}): sbatch {sbatch_script_path.resolve()}"
        )
        try:
            subprocess.run(["sbatch", str(sbatch_script_path.resolve())], check=True)
            print(f"  Successfully submitted job {job_actual_name}.")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR submitting job {job_actual_name}: {e}")
            print(f"  The sbatch script is saved at: {sbatch_script_path}")
            print(
                f"  Please inspect its content, e.g., with 'cat {sbatch_script_path}' or 'nano {sbatch_script_path}'"
            )
        except FileNotFoundError:
            sys.exit(
                "ERROR: sbatch command not found. Ensure Slurm tools are in your PATH."
            )
        time.sleep(0.1)

    if args.dry_run:
        print(
            f"\n(Dry run complete – nothing was submitted. Scripts are saved in {sbatch_scripts_dir})"
        )
    else:
        print(
            f"\n✅ All {len(experiment_configs)} jobs processed. Check Slurm queue (`squeue -u $USER`)."
        )
        print(f"Output and logs will be in subdirectories of: {root}")
        print(f"Submitted sbatch scripts are in: {sbatch_scripts_dir}")


if __name__ == "__main__":

    main()
