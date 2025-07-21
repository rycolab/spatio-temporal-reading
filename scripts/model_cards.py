from pathlib import Path
from re import L

from matplotlib.dates import MO
import test
from torchaudio import load
from model_checkpoints import *


# dir_exps = Path(__file__).parent.parent / "cluster_runs" / "saccade"
dir_exps = Path(__file__).parent.parent / "local_runs" / "saccade"

# set subset to true if you want to quickly test the models on a small subset of the data (e.g. 2000 samples, you can change this in config.py)

# We write "true" or "false" as strings, so that we can use them as CLI arguments
subset = "true"  # "true" | "false"
load_checkpoint = "false"  # "true" | "false"
training = "false"  # "true" | "false"
testing = "true"  # "true" | "false"


added_dict = {
    "subset": subset,
    "training": training,
    "final_testing": testing,
    "experiment_dir": str(dir_exps),
}

# set load_checkpoint to "false" if you want to train the model from scratch, "true" if you want to load a pre-trained model
# we load pre-trained models from the dictionaries CHECKPOINT_SACCADE_MARKS and CHECKPOINT_DURATION_MARKS, which are defined in model_checkpoints.py
# and link to the best models of the respective runs. The checkpoints are not stored in the repository for size reasons,
# but you can email us to have them sent to you.

# *****************#
# * Saccade Models #
# *****************#
missing_value_term = "linear_term"
POISSON_PROCESS_RAW = dict(
    model_type="saccade",
    saccade_likelihood="HomogenousPoisson",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    saccade_predictors_funcs="past_position",
    directory_name="poisson_raw_baseline",
    test_model_dir=None,
    batch_size=512,
    learning_rate=0.001,
    load_checkpoint="false",
)

POISSON_PROCESS_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="HomogenousPoisson",
    dataset_filtering="filtered",
    missing_value_effects=missing_value_term,
    saccade_predictors_funcs="past_position",
    directory_name="poisson_filtered_baseline",
    test_model_dir=None,
    batch_size=512,
    learning_rate=0.001,
    load_checkpoint="false",
)

LAST_FIXATION_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="LastFixationModel",
    saccade_predictors_funcs="past_position",
    directory_name="BASE_LF_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    load_checkpoint="false",
)

STANDARD_HAWKES_PROCESS_RAW = dict(
    model_type="saccade",
    saccade_likelihood="StandardHawkesProcess",
    saccade_predictors_funcs="past_position",
    directory_name="BASE_SHP_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    load_checkpoint="false",
)


CONSTANT_SPATIAL_SHIFT_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position",
    directory_name="CSS_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    load_checkpoint="false",
)


READER_MIXED_EFFECT_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader",
    directory_name="RME_CSS_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=False,
)

CHARACTER_SURPRISAL_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_char",
    directory_name="RME_CSS_CS_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["cs_raw"],
)

WORD_SURPRISAL_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_word",
    directory_name="RME_CSS_WS_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["ws_raw"],
    load_checkpoint=load_checkpoint,
)


LEN_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_len",
    directory_name="RME_CSS_LEN_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["len_raw"],
    load_checkpoint=load_checkpoint,
)

FREQ_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_freq",
    directory_name="RME_CSS_FREQ_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["freq_raw"],
    load_checkpoint=load_checkpoint,
)

RME_CSS_DURATION_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_duration",
    directory_name="RME_CSS_DUR_RAW",
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["dur_raw"],
    load_checkpoint=load_checkpoint,
)

LAST_FIXATION_MODEL_FILTERED_SCANPATH = dict(
    model_type="saccade",
    saccade_likelihood="LastFixationModel",
    saccade_predictors_funcs="past_position",
    directory_name="BASE_LF_FILTERED",
    dataset_filtering="filtered",
    load_checkpoint="false",
)


STANDARD_HAWKES_PROCESS_FILTERED_SCANPATH = dict(
    model_type="saccade",
    saccade_likelihood="StandardHawkesProcess",
    saccade_predictors_funcs="past_position",
    directory_name="BASE_SHP_FILTERED",
    dataset_filtering="filtered",
    load_checkpoint="false",
)


CONSTANT_SPATIAL_SHIFT_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position",
    directory_name="CSS_FILTERED",
    dataset_filtering="filtered",
    load_checkpoint="false",
)

READER_MIXED_EFFECT_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader",
    directory_name="RME_CSS_FILTERED",
    dataset_filtering="filtered",
    load_checkpoint="false",
)

CHARACTER_SURPRISAL_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_char",
    directory_name="RME_CSS_CS_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["cs_filtered"],
    load_checkpoint=load_checkpoint,
)

WORD_SURPRISAL_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_word",
    directory_name="RME_CSS_WS_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["ws_filtered"],
    load_checkpoint=load_checkpoint,
)
LEN_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_len",
    directory_name="RME_CSS_LEN_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["len_filtered"],
    load_checkpoint=load_checkpoint,
)
FREQ_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_freq",
    directory_name="RME_CSS_FREQ_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["freq_filtered"],
    load_checkpoint=load_checkpoint,
)

RME_CSS_DURATION_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_duration",
    directory_name="RME_CSS_DUR_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=CHECKPOINT_SACCADE_MARKS["dur_filtered"],
    load_checkpoint=load_checkpoint,
)


LEN_FREQ_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_len_freq",
    directory_name="RME_CSS_LEN_FREQ_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=sacc_filtered_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

CHAR_LEN_FREQ_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_char_len_freq",
    directory_name="RME_CSS_CHAR_LEN_FREQ_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=sacc_filtered_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

WORD_LEN_FREQ_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_word_len_freq",
    directory_name="RME_CSS_WORD_LEN_FREQ_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=sacc_filtered_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

LEN_FREQ_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_len_freq",
    directory_name="RME_CSS_LEN_FREQ_RAW",
    dataset_filtering="raw",
    checkpoint_path=sacc_raw_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

CHAR_LEN_FREQ_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_char_len_freq",
    directory_name="RME_CSS_CHAR_LEN_FREQ_RAW",
    dataset_filtering="raw",
    checkpoint_path=sacc_raw_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

WORD_LEN_FREQ_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_word_len_freq",
    directory_name="RME_CSS_WORD_LEN_FREQ_RAW",
    dataset_filtering="raw",
    checkpoint_path=sacc_raw_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

CHAR_WORD_LEN_FREQ_RME_CSS_MODEL_RAW = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_char_word_len_freq",
    directory_name="RME_CSS_CHAR_WORD_LEN_FREQ_RAW",
    dataset_filtering="raw",
    checkpoint_path=sacc_raw_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

CHAR_WORD_LEN_FREQ_RME_CSS_MODEL_FILTERED = dict(
    model_type="saccade",
    saccade_likelihood="ExtendedHawkesProcess",
    saccade_predictors_funcs="past_position_reader_char_word_len_freq",
    directory_name="RME_CSS_CHAR_WORD_LEN_FREQ_FILTERED",
    dataset_filtering="filtered",
    checkpoint_path=sacc_filtered_rme_css_model_best_path,
    load_checkpoint=load_checkpoint,
)

# ******************#
# * Duration Models #
# ******************#
division_factor_durations = 1

# --- Unfiltered / raw --------------------------------------------------------
DURATION_BASELINE_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    duration_predictors_funcs="dur_model_baseline",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=False,
    directory_name="dur_baseline_raw",
    test_model_dir=None,
    batch_size=512,
    learning_rate=0.001,
    load_checkpoint="false",
)

DURATION_RME_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_RAW",
    duration_predictors_funcs="dur_model_reader_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path="None",
)

DURATION_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_DUR_RAW",
    duration_predictors_funcs="dur_model_reader_dur_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_raw_rme_path,
)

CHARACTER_SURPRISAL_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_CS_RAW",
    duration_predictors_funcs="dur_model_reader_char_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_raw_rme_path,
)

WORD_SURPRISAL_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_WS_RAW",
    duration_predictors_funcs="dur_model_reader_word_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_raw_rme_path,
)

FREQ_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_FREQ_RAW",
    duration_predictors_funcs="dur_model_reader_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
)


LEN_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_LEN_RAW",
    duration_predictors_funcs="dur_model_reader_len_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
)

LEN_FREQ_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_LEN_FREQ_RAW",
    duration_predictors_funcs="dur_model_reader_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
)

CHAR_LEN_FREQ_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_CHAR_LEN_FREQ_RAW",
    duration_predictors_funcs="dur_model_reader_char_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
)

WORD_LEN_FREQ_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_WORD_LEN_FREQ_RAW",
    duration_predictors_funcs="dur_model_reader_word_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
)

CHAR_WORD_LEN_FREQ_RME_DUR_MODEL_RAW = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_CHAR_WORD_LEN_FREQ_RAW",
    duration_predictors_funcs="dur_model_reader_char_word_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="raw",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
)


# --- Filtered ---------------------------------------------------------------
DURATION_BASELINE_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    duration_predictors_funcs="dur_model_baseline",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    checkpoint_path=dur_filtered_rme_path,
    directory_name="duration_baseline_filtered",
    test_model_dir=None,
    batch_size=512,
    load_checkpoint="false",
)

DURATION_RME_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_FILTERED",
    duration_predictors_funcs="dur_model_reader_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    checkpoint_path="None",
    load_checkpoint="false",
)

DURATION_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_DUR_FILTERED",
    duration_predictors_funcs="dur_model_reader_dur_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)

CHARACTER_SURPRISAL_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_CS_FILTERED",
    duration_predictors_funcs="dur_model_reader_char_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)

WORD_SURPRISAL_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_WS_FILTERED",
    duration_predictors_funcs="dur_model_reader_word_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)


FREQ_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_FREQ_FILTERED",
    duration_predictors_funcs="dur_model_reader_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)
LEN_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_LEN_FILTERED",
    duration_predictors_funcs="dur_model_reader_len_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)

LEN_FREQ_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_LEN_FREQ_FILTERED",
    duration_predictors_funcs="dur_model_reader_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)

CHAR_LEN_FREQ_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_CHAR_LEN_FREQ_FILTERED",
    duration_predictors_funcs="dur_model_reader_char_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)

WORD_LEN_FREQ_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_WORD_LEN_FREQ_FILTERED",
    duration_predictors_funcs="dur_model_reader_word_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)


CHAR_WORD_LEN_FREQ_RME_DUR_MODEL_FILTERED = dict(
    model_type="duration",
    dur_likelihood="normal",
    directory_name="RME_LN_CHAR_WORD_LEN_FREQ_FILTERED",
    duration_predictors_funcs="dur_model_reader_char_word_len_freq_conv_features",
    division_factor_durations=division_factor_durations,
    dataset_filtering="filtered",
    missing_value_effects=missing_value_term,
    checkpoint_path=dur_filtered_rme_path,
    load_checkpoint=load_checkpoint,
)

MODELS = dict(
    # --- Saccade models on raw scanpaths
    poisson_raw=POISSON_PROCESS_RAW,
    last_fix_raw=LAST_FIXATION_MODEL_RAW,
    stand_hawkes_raw=STANDARD_HAWKES_PROCESS_RAW,
    css_raw=CONSTANT_SPATIAL_SHIFT_MODEL_RAW,
    rme_css_raw=READER_MIXED_EFFECT_CSS_MODEL_RAW,
    rme_css_cs_raw=CHARACTER_SURPRISAL_RME_CSS_MODEL_RAW,
    rme_css_ws_raw=WORD_SURPRISAL_RME_CSS_MODEL_RAW,
    rme_css_dur_raw=RME_CSS_DURATION_MODEL_RAW,
    rme_css_len_raw=LEN_RME_CSS_MODEL_RAW,
    rme_css_freq_raw=FREQ_RME_CSS_MODEL_RAW,
    rme_css_len_freq_raw=LEN_FREQ_RME_CSS_MODEL_RAW,
    rme_css_char_len_freq_raw=CHAR_LEN_FREQ_RME_CSS_MODEL_RAW,
    rme_css_word_len_freq_raw=WORD_LEN_FREQ_RME_CSS_MODEL_RAW,
    rme_css_char_word_len_freq_raw=CHAR_WORD_LEN_FREQ_RME_CSS_MODEL_RAW,
    # --- Saccade models on filtered scanpaths
    poisson_filtered=POISSON_PROCESS_FILTERED,
    last_fix_filtered=LAST_FIXATION_MODEL_FILTERED_SCANPATH,
    stand_hawkes_filtered=STANDARD_HAWKES_PROCESS_FILTERED_SCANPATH,
    css_filtered=CONSTANT_SPATIAL_SHIFT_MODEL_FILTERED,
    rme_css_filtered=READER_MIXED_EFFECT_CSS_MODEL_FILTERED,
    rme_css_cs_filtered=CHARACTER_SURPRISAL_RME_CSS_MODEL_FILTERED,
    rme_css_ws_filtered=WORD_SURPRISAL_RME_CSS_MODEL_FILTERED,
    rme_css_dur_filtered=RME_CSS_DURATION_MODEL_FILTERED,
    rme_css_len_filtered=LEN_RME_CSS_MODEL_FILTERED,
    rme_css_freq_filtered=FREQ_RME_CSS_MODEL_FILTERED,
    rme_css_len_freq_filtered=LEN_FREQ_RME_CSS_MODEL_FILTERED,
    rme_css_char_len_freq_filtered=CHAR_LEN_FREQ_RME_CSS_MODEL_FILTERED,
    rme_css_word_len_freq_filtered=WORD_LEN_FREQ_RME_CSS_MODEL_FILTERED,
    rme_css_char_word_len_freq_filtered=CHAR_WORD_LEN_FREQ_RME_CSS_MODEL_FILTERED,
    # --- Duration models
    # Duration – raw
    dur_baseline_raw=DURATION_BASELINE_RAW,
    dur_rme_raw=DURATION_RME_MODEL_RAW,
    dur_rme_dur_raw=DURATION_RME_DUR_MODEL_RAW,
    dur_rme_cs_raw=CHARACTER_SURPRISAL_RME_DUR_MODEL_RAW,
    dur_rme_ws_raw=WORD_SURPRISAL_RME_DUR_MODEL_RAW,
    dur_rme_freq_raw=FREQ_RME_DUR_MODEL_RAW,
    dur_rme_len_raw=LEN_RME_DUR_MODEL_RAW,
    dur_rme_len_freq_raw=LEN_FREQ_RME_DUR_MODEL_RAW,
    dur_rme_char_len_freq_raw=CHAR_LEN_FREQ_RME_DUR_MODEL_RAW,
    dur_rme_word_len_freq_raw=WORD_LEN_FREQ_RME_DUR_MODEL_RAW,
    dur_rme_char_word_len_freq_raw=CHAR_WORD_LEN_FREQ_RME_DUR_MODEL_RAW,
    # Duration – filtered
    dur_baseline_filtered=DURATION_BASELINE_FILTERED,
    dur_rme_filtered=DURATION_RME_MODEL_FILTERED,
    dur_rme_dur_filtered=DURATION_RME_DUR_MODEL_FILTERED,
    dur_rme_cs_filtered=CHARACTER_SURPRISAL_RME_DUR_MODEL_FILTERED,
    dur_rme_ws_filtered=WORD_SURPRISAL_RME_DUR_MODEL_FILTERED,
    dur_rme_freq_filtered=FREQ_RME_DUR_MODEL_FILTERED,
    dur_rme_len_filtered=LEN_RME_DUR_MODEL_FILTERED,
    dur_rme_len_freq_filtered=LEN_FREQ_RME_DUR_MODEL_FILTERED,
    dur_rme_char_len_freq_filtered=CHAR_LEN_FREQ_RME_DUR_MODEL_FILTERED,
    dur_rme_word_len_freq_filtered=WORD_LEN_FREQ_RME_DUR_MODEL_FILTERED,
    dur_rme_char_word_len_freq_filtered=CHAR_WORD_LEN_FREQ_RME_DUR_MODEL_FILTERED,
)


MODELS = {k: {**v, **added_dict} for k, v in MODELS.items()}
