#!/usr/bin/env python3
"""
run_test_eval_global.py
======================

This script scans all *immediate* sub‑directories of a user‑supplied
root directory.  Whenever it finds a folder structure like

    <experiment_dir>/best_model/train.log

**and** the experiment does *not* already contain a directory whose name
starts with ``test_set_eval`` or ``test_eval`` (indicating evaluation
done earlier), it reads the first parameter dictionary in
``train.log``, infers whether the experiment trained a **saccade** or
**duration** model, and then launches ``main.py`` once per experiment in
final‑testing mode with the same hyper‑parameters that produced the best
checkpoint.

Output from each evaluation is streamed live.  The script finishes with
a summary and exits non‑zero if any individual evaluation fails.

Usage
-----
::

    python run_test_eval_global.py /path/to/experiments_root

Expected tree::

    experiments_root/
    ├── exp_01/
    │   └── best_model/train.log   # evaluated
    ├── exp_02/                    # skipped – no best_model
    ├── exp_03/
    │   ├── best_model/train.log
    │   └── test_set_eval/         # skipped – already evaluated
    └── exp_04/                    # skipped – malformed log

"""
from __future__ import annotations

import argparse
import ast
import json
import pdb
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_dict_string(dict_str: str) -> Dict[str, Any]:
    """Parse ``dict_str`` as JSON, falling back to ``ast.literal_eval``."""
    try:
        return json.loads(dict_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(dict_str)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Cannot parse as JSON or Python literal: " f"{dict_str[:80]}…"
            ) from exc


def extract_params_dict(train_log: Path) -> Dict[str, Any]:
    """Return the *first* dictionary literal / JSON object found in *train_log*."""
    text = train_log.read_text(encoding="utf-8")
    matches = re.findall(r"(\{.*?\})", text, flags=re.DOTALL)
    if not matches:
        raise RuntimeError("No dictionary literals found in train.log")
    return parse_dict_string(matches[0])


def build_command(
    project_root: Path, best_model_dir: Path, params: Dict[str, Any]
) -> List[str]:
    """Construct the ``python -u main.py …`` command for final testing."""
    main_py = project_root / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"main.py not found at {main_py}")

    model_type = params.get("model_type")
    if model_type not in {"saccade", "duration"}:
        raise ValueError(f"Unsupported model_type '{model_type}'")
    params.pop("final_testing")
    params.pop("training")
    params.pop("subset")
    params.pop("load_checkpoint")
    params.pop("experiment_dir")
    params.pop("directory_name")
    params.pop("seed")
    params.pop("strict_load")
    params.pop("gradient_clipping")
    params.pop("test_model_dir")
    cmd: List[str] = [
        sys.executable,
        "-u",
        str(main_py),
        "--final-testing",
        "true",
        "--training",
        "false",
        "--subset",
        "false",
        "--load-checkpoint",
        "false",
        "--test-model-dir",
        str(best_model_dir),
        "--experiment-dir",
        str(best_model_dir.parent),
        "--directory-name",
        "test_set_eval",
        "--seed",
        "8045",
    ]

    for key, val in params.items():
        flag = f"--{key.replace('_', '-')}"
        cmd.extend([flag, str(val)])

    return cmd


# ---------------------------------------------------------------------------
# Execution logic
# ---------------------------------------------------------------------------


def run_evaluation(exp_dir: Path) -> int:
    """Run evaluation for *exp_dir*. Returns the exit code."""
    best_model_dir = exp_dir / "best_model"
    train_log = best_model_dir / "train.log"

    params = extract_params_dict(train_log)
    project_root = Path(__file__).resolve().parent.parent
    command = build_command(project_root, best_model_dir, params)

    print(f"\n=== Evaluating {exp_dir.name} ({params['model_type']}) ===")
    print("Command:", " ".join(command))
    proc = subprocess.Popen(
        command,
        cwd=project_root,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        for line in proc.stdout:  # type: ignore[assignment]
            print(line, end="")
    finally:
        proc.stdout.close()  # type: ignore[arg-type]

    return proc.wait()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final testing for every experiment under <root_dir>."
    )
    parser.add_argument("root_dir", type=Path, help="Parent directory of experiments")
    args = parser.parse_args()

    root = args.root_dir.resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    EVAL_DIR_PREFIXES = ("test_set_eval", "test_eval")

    experiments: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        best_model_dir = child / "best_model"
        train_log = best_model_dir / "train.log"
        if not train_log.is_file():
            print(f"Skipping {child.name}: no best_model/train.log found.")
            continue

        # Skip if evaluation already present
        already_evaluated = any(
            d.is_dir() and any(d.name.startswith(p) for p in EVAL_DIR_PREFIXES)
            for d in child.iterdir()
        )
        if already_evaluated:
            print(f"Skipping {child.name}: evaluation directory already exists.")
            continue

        experiments.append(child)

    if not experiments:
        print("No experiments require evaluation.  Nothing to do.")
        return

    failures = 0
    for exp in experiments:
        try:
            rc = run_evaluation(exp)
            if rc == 0:
                print(f"✔ {exp.name}: success\n")
            else:
                failures += 1
                print(f"✗ {exp.name}: exited with code {rc}\n", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"✗ {exp.name}: exception {exc}\n", file=sys.stderr)

    if failures:
        print(f"Finished with {failures} failed run(s).", file=sys.stderr)
        sys.exit(1)
    else:
        print("All experiments evaluated successfully.")


if __name__ == "__main__":
    main()
