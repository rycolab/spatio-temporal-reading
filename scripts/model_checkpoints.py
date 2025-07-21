from pathlib import Path
import pdb

from pyparsing import C

# if false the checkpoint will be loaded from the best existing checkpoing of the current model
CHECKPOINT_FROM_SIMPLER_MODEL = True
experiment_folder = Path(__file__).parent.parent / "cluster_runs"

if not experiment_folder.exists():
    experiment_folder.mkdir(parents=True, exist_ok=True)
    (experiment_folder / "saccade").mkdir(parents=True, exist_ok=True)
    (experiment_folder / "duration").mkdir(parents=True, exist_ok=True)


# *******************#
# * SACCADE MODELS    #
# *********************


sacc_filtered_rme_css_model_best_path = str(
    experiment_folder
    / "saccade"
    / "rme_css_filtered_2025-06-04_19-43-59-222"
    / "best_model"
)
sacc_raw_rme_css_model_best_path = str(
    experiment_folder / "saccade" / "rme_css_raw_2025-06-04_19-42-36-801" / "best_model"
)

marks = ["dur", "freq", "len", "ws", "cs_"]
dataset_types = ["filtered", "raw"]
marks_to_check_against = set(["dur", "freq", "len", "word", "char", "ws", "cs"])

CHECKPOINT_SACCADE_MARKS = {}

dir_names = list((experiment_folder / "saccade").iterdir())

valid_dir_names = []
for name in dir_names:
    atomic_names_set = str(name).split("_")

    # check how many marks are contained in the atomic names
    atomic_names_set = set(atomic_names_set)
    atomic_names_set = atomic_names_set.intersection(marks_to_check_against)
    if len(atomic_names_set) == 1:
        valid_dir_names.append(name)
for mark in marks:

    potential_paths = [name for name in valid_dir_names if mark in str(name)]
    for dataset_type in dataset_types:
        path = [name for name in potential_paths if dataset_type in str(name)]
        mark = mark.strip("_")

        if CHECKPOINT_FROM_SIMPLER_MODEL:
            if dataset_type == "filtered":
                CHECKPOINT_SACCADE_MARKS[f"{mark}_{dataset_type}"] = (
                    sacc_filtered_rme_css_model_best_path
                )
            elif dataset_type == "raw":
                CHECKPOINT_SACCADE_MARKS[f"{mark}_{dataset_type}"] = (
                    sacc_raw_rme_css_model_best_path
                )
        else:
            if len(path) == 1:
                CHECKPOINT_SACCADE_MARKS[f"{mark}_{dataset_type}"] = str(
                    path[0] / "best_model"
                )
            else:
                CHECKPOINT_SACCADE_MARKS[f"{mark}_{dataset_type}"] = None

# ********************#
# * DURATION MODELS    #
# **********************

dur_filtered_rme_path = str(
    experiment_folder
    / "duration"
    / "dur_rme_filtered_2025-06-06_02-56-22-332"
    / "best_model"
)

dur_raw_rme_path = str(
    experiment_folder
    / "duration"
    / "dur_rme_raw_2025-06-06_02-56-22-332"
    / "best_model"
)
