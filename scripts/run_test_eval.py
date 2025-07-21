import argparse
from pathlib import Path
import pdb
import subprocess
import sys
import re
import json
import ast


def parse_dict_string(dict_str: str):
    """
    Tries to parse a string as JSON, falls back to ast.literal_eval for Python dicts.
    """
    try:
        return json.loads(dict_str)
    except json.JSONDecodeError:
        try:
            # ast.literal_eval can safely evaluate string representations of Python literals
            return ast.literal_eval(dict_str)
        except (SyntaxError, ValueError) as e:
            raise ValueError(
                f"String content '{dict_str[:100]}...' is not a valid JSON or Python dict literal."
            ) from e


def main():
    parser = argparse.ArgumentParser(
        description="Runs final testing by parsing parameters from best_model/train.log."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the 'best_model' subdirectory.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--saccade",
        action="store_true",
        help="If set, will run the saccade model. Mutually exclusive with --duration.",
    )

    group.add_argument(
        "--duration",
        action="store_true",
        help="If set, will run the duration model. Mutually exclusive with --saccade.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()  # Resolve to an absolute path

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)

    best_model_dir = input_dir / "best_model"
    if not best_model_dir.is_dir():
        print(f"Error: Directory 'best_model' not found in '{input_dir}'.")
        sys.exit(1)

    train_log_path = best_model_dir / "train.log"
    if not train_log_path.is_file():
        print(f"Error: File 'train.log' not found in '{best_model_dir}'.")
        sys.exit(1)

    current_script_path = Path(__file__).resolve()
    project_root_dir = current_script_path.parent.parent
    main_script_path = project_root_dir / "main.py"

    if not main_script_path.is_file():
        print(f"Error: Script 'main.py' not found at '{main_script_path}'.")
        sys.exit(1)

    # Read and parse train.log
    try:
        with open(train_log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading '{train_log_path}': {e}")
        sys.exit(1)

    dict_strings_found = re.findall(r"(\{.*?\})", log_content, re.DOTALL)
    if not len(dict_strings_found):
        print(
            f"Error: No dictionary strings found in '{train_log_path}'. "
            "Ensure the log contains valid JSON or Python dict strings."
        )
        sys.exit(1)

    try:
        # Parse the first two found dictionary strings
        dict_str = dict_strings_found[0]

        params_dict = parse_dict_string(dict_str)

    except ValueError as e:
        print(f"Error parsing dictionary strings from '{train_log_path}':")
        print(e)
        if "dict_str" in locals() and not isinstance(params_dict, dict):
            print(f"Problematic string 1 (first 100 chars): {dict_str[:100]}...")

        sys.exit(1)

    if args.saccade:
        if params_dict["model_type"] != "saccade":
            print(
                f"Error: The first dictionary string does not match the saccade model type. "
                f"Expected 'saccade', got '{params_dict['model_type']}'."
            )
            sys.exit(1)
        args_pars = dict(
            model_type=params_dict["model_type"],
            saccade_likelihood=params_dict["saccade_likelihood"],
            saccade_predictors_funcs=params_dict["saccade_predictors_funcs"],
            dataset_filtering=params_dict["dataset_filtering"],
            directory_name="test_set_eval_",
        )

    elif args.duration:
        if params_dict["model_type"] != "duration":
            print(
                f"Error: The first dictionary string does not match the duration model type. "
                f"Expected 'duration', got '{params_dict['model_type']}'."
            )
            sys.exit(1)
        args_pars = dict(
            model_type=params_dict["model_type"],
            dur_likelihood=params_dict["dur_likelihood"],
            duration_predictors_funcs=params_dict["duration_predictors_funcs"],
            directory_name="test_set_eval",
        )
    # Construct the command to run main.py
    # sys.executable ensures using the same python interpreter

    command_args_from_params_dict = []
    for key, value in args_pars.items():
        # Transform key: replace underscores with hyphens for command line flags
        arg_name = f"--{key.replace('_', '-')}"
        command_args_from_params_dict.append(arg_name)
        command_args_from_params_dict.append(str(value))

    command = [
        sys.executable,
        str(main_script_path),
        "--final-testing",
        str(True),
        "--training",
        str(False),
        "--subset",
        str(False),
        str("--load-checkpoint"),
        str(False),
        "--test-model-dir",
        str(best_model_dir),
        "--experiment-dir",
        str(best_model_dir.parent),
        "--directory-name",
        "test_set_eval",
        "--seed",
        str(8045),
    ]
    command.extend(command_args_from_params_dict)
    command.insert(1, "-u")  # becomes: [python, -u, main.py, …]

    print(f"Running command: {' '.join(command)}")

    # Start the process
    process = subprocess.Popen(
        command,
        cwd=project_root_dir,
        text=True,
        bufsize=1,  # line-buffered
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout (optional)
    )

    # Forward output live
    try:
        for line in process.stdout:  # blocks until a line is ready
            print(line, end="")  # already includes its own newline
    finally:
        process.stdout.close()

    returncode = process.wait()
    if returncode != 0:
        sys.exit(returncode)


if __name__ == "__main__":
    main()
