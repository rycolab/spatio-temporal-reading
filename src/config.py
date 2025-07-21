from __future__ import annotations
from dataclasses import asdict, fields, replace

import argparse
import json
import logging
import os
import pdb
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict
import typing
from types import (
    NoneType,
    UnionType as Python10UnionType,
)


@dataclass
class RunConfig:

    # *********************#
    # Hyperparameters
    # *********************#
    # trainer options
    epochs: int = 2
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "SGDNesterov"  # Adam | SGDNesterov
    gradient_clipping: bool = True
    patience: int = 5
    lr_rescaling: float = 0.99
    training: str = "true"  # "true" | "false"
    final_testing: str = "true"  # "true" | "false"

    # dataset
    splitting_procedure: str = "random_shuffle"
    subset: str = "true"  # "true" | "false"
    subset_size: int = 2_000

    dataset_filtering: str = "filtered"  # "filtered" | "raw"
    model_type: str = "saccade"  # "saccade" | "duration"
    missing_value_effects: str = "linear_term"  # "linear_term" | "ignore" |

    # **** Saccade Model ****
    saccade_likelihood: str = (
        "HomogenousPoisson"  # "HomogenousPoisson" | "StandardHawkesProcess" | "ExtendedHawkesProcess", "LastFixationModel"
    )
    saccade_predictors_funcs: str = "past_position"
    # "past_position" | "past_position_reader" |
    # "past_position_reader_duration" | "past_position_reader_char" | "past_position_reader_word"

    # **** Duration Model ****
    dur_likelihood: str = (
        "normal"  # "rayleigh" | "exponential" | "lognormal" | "normal"
    )
    # Ensure type hint for fields without one, if they are to be CLI args
    duration_predictors_funcs: str = "dur_model_reader_dur_conv_features"
    # dur_model_baseline
    # dur_model_reader_char_conv_features, dur_model_reader_dur_conv_features, dur_model_reader_word_conv_features

    # model loading
    load_checkpoint: str = "false"
    checkpoint_path: Path | None = None
    strict_load: bool = False

    # reproducibility / hardware
    seed: int = 124
    nworkers: int = 0

    # directory for the experiments
    experiment_dir: str = str("runs")
    # directory for the specific run
    directory_name: str = f"hp_{saccade_likelihood}"

    # we set this to None whenever we want to test on the same model we are training
    # if we set testing = True, training = False, we will not train a model, so we can set this to the directory of the model we want to test
    test_model_dir: Path | None = (
        "/Users/francescoignaziore/Projects/fine-grained-model-reading-behaviour/cluster_runs/saccade/rme_css_len_raw_2025-06-06_05-12-54-487/best_model"
    )
    # In the Meco dataset durations and saccades are expressed in milliseconds
    # interarrival saccades times between two consequent saccades have a median of 27 ms.
    # in an exponential kernel, a * exp-b(27) is an extremely small value unless there is a big value of (a,b) to counterbalance it.
    # In order to avoid numerical issues, we divide the saccade-intervals by 1000 to convert them to seconds, to allow for a range of values of plausible candidate (a,b) that is more stable for optimization.

    # scaling factors to avoid numerical issues
    division_factor_space: int = 100
    division_factor_time: int = 1000
    division_factor_durations: int = 1

    past_timesteps_duration_baseline_k: int = 10

    # initialization of parameters for convolution gamma kernel
    alpha_g: float = 0.1
    delta_g: float = 0.1
    beta_g: float = 0.1

    @staticmethod
    def from_cli() -> "RunConfig":
        cfg = RunConfig()  # Dataclass instance with defaults
        parser = argparse.ArgumentParser()
        # Resolve stringified type hints (due to "from __future__ import annotations")
        # to actual type objects.
        resolved_types = typing.get_type_hints(RunConfig)

        for f_desc in fields(RunConfig):  # f_desc is a dataclasses.Field object

            flag_name = f"--{f_desc.name.replace('_','-')}"

            actual_field_type = resolved_types[f_desc.name]

            # Use f_desc.default if available and not MISSING, otherwise getattr.
            # Since all fields here have defaults, f_desc.default should be fine.
            default_value = f_desc.default
            if actual_field_type is bool:
                parser.add_argument(
                    flag_name,
                    dest=f_desc.name,
                    action="store_true",
                    default=argparse.SUPPRESS,  # <-- omit if not present
                    help=f"Disable {f_desc.name}",
                )

            # Boolean flags: store_true if default is False, store_false if default is True
            # if default_value is True:
            #    parser.add_argument(
            #        flag_name, dest=f_desc.name, action="store_false"
            #    )
            # else:  # default_value is False

            #    parser.add_argument(
            #        flag_name, dest=f_desc.name, action="store_true"
            #    )
            # Argparse handles default for store_true/false actions automatically.
            # Setting default explicitly can be redundant or conflict.
            else:
                # Process type for argparse (handles Optional[X], etc.)
                cli_arg_type = _cli_type(actual_field_type)

                parser.add_argument(
                    flag_name,
                    dest=f_desc.name,
                    type=cli_arg_type,
                    default=default_value,
                )

        parsed_args_ns = parser.parse_args()

        # Create a new config instance updated with parsed CLI arguments
        # vars(parsed_args_ns) will have values for all fields (either from CLI or their defaults)
        updated_cfg = replace(cfg, **vars(parsed_args_ns))

        objs = convert_str_to_bool(
            updated_cfg,
            ["load_checkpoint", "training", "final_testing", "subset"],
        )
        updated_cfg = replace(cfg, **vars(objs))

        return updated_cfg


def _cli_type(t: Any) -> type | None:
    """
    Process a type hint for argparse.
    Handles Optional[X] (X | None) by returning X.
    Returns the type if it's directly callable by argparse (like int, str, Path).
    """
    # Check for Python 3.10+ UnionType (e.g., int | None)
    is_pep604_union = isinstance(t, Python10UnionType)
    # Check for typing.Union (e.g., typing.Union[int, None])
    is_typing_union = hasattr(t, "__origin__") and t.__origin__ is typing.Union

    if is_pep604_union or is_typing_union:
        args = t.__args__
        # Filter out NoneType
        non_none_args = [
            arg for arg in args if arg is not NoneType and arg is not type(None)
        ]  # type(None) for older Pythons
        if len(non_none_args) == 1:
            # If it was Optional[X], return X
            # Ensure X itself is callable (e.g. int, str, Path)
            if callable(non_none_args[0]):
                return non_none_args[0]
            else:
                raise TypeError(
                    f"Union argument type {non_none_args[0]!r} is not callable."
                )
        elif len(non_none_args) > 1:
            # Unions like int | str are not directly supported by basic argparse type conversion
            raise TypeError(
                f"Multi-argument Union {t!r} not supported for CLI parsing. Use a custom type function."
            )
        else:  # Only NoneType, or empty union
            raise TypeError(
                f"Union {t!r} does not yield a suitable type for CLI parsing."
            )

    # For non-Union types, check if callable (e.g. int, str, Path)
    if callable(t):
        return t

    # If 't' is not a recognized Union and not callable, it's an issue.
    # This might happen if a type hint string couldn't be resolved by get_type_hints correctly.
    raise TypeError(
        f"Type {t!r} is not callable or a supported Union type for CLI parsing."
    )


from dataclasses import asdict
from argparse import Namespace


def convert_str_to_bool(config_obj, bool_string_keys):

    cfg_dict = asdict(config_obj)

    for key in bool_string_keys:
        val = cfg_dict.get(key)
        if isinstance(val, str):
            if val == "true":
                cfg_dict[key] = True
            elif val == "false":
                cfg_dict[key] = False
            else:
                raise ValueError(f"Key '{key}' has invalid string for boolean: {val!r}")
        elif isinstance(val, bool):
            continue
        else:
            raise TypeError(
                f"Key '{key}' must be a string 'True'/'False' or bool, but got {type(val)}"
            )

    return Namespace(**cfg_dict)
