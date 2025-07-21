import argparse
from pathlib import Path
import subprocess
import sys
from model_cards import MODELS


def main():
    """Parses CLI args and executes main.py with the selected configuration."""
    parser = argparse.ArgumentParser(
        description="Run a predefined model configuration from main.py.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)

    # Dynamically create a flag for each model in the MODELS dictionary
    help_text_lines = ["Available models:"]
    for name in MODELS.keys():
        flag = f"--{name.replace('_', '-')}"
        help_text_lines.append(f"  {flag}")
        group.add_argument(
            flag,
            dest=name,  # Store in a destination with the original underscore name
            action="store_true",
            help=f"Run the '{name}' configuration.",
        )
    parser.epilog = "\n".join(help_text_lines)

    args = parser.parse_args()

    # Find which model flag was set by the user
    selected_model_name = None
    for name, was_set in vars(args).items():
        if was_set:
            selected_model_name = name
            break

    # This should not happen due to the 'required=True' group, but as a safeguard:
    if not selected_model_name:
        print("Error: A model flag must be provided.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Get the corresponding parameter dictionary
    params = MODELS[selected_model_name]
    print(f"--- Running Model: {selected_model_name} ---")

    # Construct the command to run main.py
    # Assumes main.py is in the same directory as this script.
    main_script_path = Path(__file__).parent.parent / "main.py"
    command = [sys.executable, str(main_script_path)]

    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        command.append(flag)
        # Add the value. The `main.py` script is set up to parse these strings.
        command.append(str(value))

    # Print and execute the command
    print("--- Executing Command ---")
    # Use a helper function to print the command in a copy-paste friendly format
    print(" ".join(f'"{c}"' if " " in c else c for c in command))
    print("---------------------------\n", flush=True)

    try:
        # Use subprocess.run to execute the command and stream output
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n--- Error running main.py ---", file=sys.stderr)
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"\n--- Error ---", file=sys.stderr)
        print(f"Could not find '{main_script_path}'.", file=sys.stderr)
        print(
            "Make sure 'run_model.py' is in the same directory as 'main.py'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n--- Script finished successfully ---")


if __name__ == "__main__":
    main()
