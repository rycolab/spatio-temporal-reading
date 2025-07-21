import numpy as np
import pandas as pd
import torch


def get_features_func(predictor_function):

    # *** DURATION MODEL PREDICTORS ***#

    if predictor_function == "dur_model_baseline":
        return dur_model_baseline
    elif predictor_function == "dur_model_reader_features":
        return dur_model_reader_features
    elif predictor_function == "dur_model_reader_dur_conv_features":
        return dur_model_reader_dur_conv_features
    elif predictor_function == "dur_model_reader_char_conv_features":
        return dur_model_reader_char_conv_features
    elif predictor_function == "dur_model_reader_word_conv_features":
        return dur_model_reader_word_conv_features
    elif predictor_function == "dur_model_reader_len_conv_features":
        return dur_model_reader_len_conv_features
    elif predictor_function == "dur_model_reader_freq_conv_features":
        return dur_model_reader_freq_conv_features
    elif predictor_function == "dur_model_reader_len_freq_conv_features":
        return dur_model_reader_len_freq_conv_features
    elif predictor_function == "dur_model_reader_char_len_freq_conv_features":
        return dur_model_reader_char_len_freq_conv_features
    elif predictor_function == "dur_model_reader_word_len_freq_conv_features":
        return dur_model_reader_word_len_freq_conv_features
    elif predictor_function == "dur_model_reader_char_word_len_freq_conv_features":
        return dur_model_reader_char_word_len_freq_features

    # *** SACCADE MODEL PREDICTORS ***#

    elif predictor_function == "past_position":
        return past_position_features
    elif predictor_function == "past_position_reader":
        return past_position_features_reader
    elif predictor_function == "past_position_reader_duration":
        return past_position_features_reader_duration
    elif predictor_function == "past_position_reader_char":
        return past_position_features_reader_char
    elif predictor_function == "past_position_reader_word":
        return past_position_features_reader_word
    elif predictor_function == "past_position_reader_freq":
        return past_position_features_reader_freq
    elif predictor_function == "past_position_reader_len":
        return past_position_features_reader_len
    elif predictor_function == "past_position_reader_len_freq":
        return past_position_features_reader_len_freq
    elif predictor_function == "past_position_reader_char_len_freq":
        return past_position_features_reader_char_len_freq
    elif predictor_function == "past_position_reader_word_len_freq":
        return past_position_features_reader_word_len_freq
    elif predictor_function == "past_position_reader_char_word_len_freq":
        return past_position_features_reader_char_word_len_freq

    else:
        raise ValueError(f"Predictor function {predictor_function} not recognized")


# *****************************#
# * SACCADE MODEL PREDICTORS  *#
# *****************************#


def past_position_features_reader_len_freq(subset, char_df, reader_to_idx):
    return past_position_features(
        subset, char_df, reader_to_idx, reader=True, freq=True, length=True
    )


def past_position_features_reader_char_word_len_freq(subset, char_df, reader_to_idx):
    return past_position_features(
        subset,
        char_df,
        reader_to_idx,
        reader=True,
        freq=True,
        char_level_surp=True,
        word_level_surp=True,
        length=True,
    )


def past_position_features_reader_char_len_freq(subset, char_df, reader_to_idx):
    return past_position_features(
        subset,
        char_df,
        reader_to_idx,
        reader=True,
        freq=True,
        char_level_surp=True,
        length=True,
    )


def past_position_features_reader_word_len_freq(subset, char_df, reader_to_idx):
    return past_position_features(
        subset,
        char_df,
        reader_to_idx,
        reader=True,
        freq=True,
        length=True,
        word_level_surp=True,
    )


def past_position_features_reader_freq(subset, char_df, reader_to_idx):
    return past_position_features(
        subset, char_df, reader_to_idx, reader=True, freq=True
    )


def past_position_features_reader_len(subset, char_df, reader_to_idx):
    return past_position_features(
        subset, char_df, reader_to_idx, reader=True, length=True
    )


def past_position_features_reader(subset, char_df, reader_to_idx):
    return past_position_features(subset, char_df, reader_to_idx, reader=True)


def past_position_features_reader_duration(subset, char_df, reader_to_idx):
    return past_position_features(
        subset, char_df, reader_to_idx, reader=True, durations=True
    )


def past_position_features_reader_char(subset, char_df, reader_to_idx):
    return past_position_features(
        subset,
        char_df,
        reader_to_idx,
        reader=True,
        durations=False,
        char_level_surp=True,
    )


def past_position_features_reader_word(subset, char_df, reader_to_idx):
    return past_position_features(
        subset,
        char_df,
        reader_to_idx,
        reader=True,
        durations=False,
        word_level_surp=True,
    )


def past_position_features(
    subset,
    char_df,
    reader_to_idx,
    reader=False,
    durations=False,
    char_level_surp=False,
    word_level_surp=False,
    freq=False,
    length=False,
):

    subset = subset.copy()
    result = subset[["x", "y"]].copy()

    if reader:
        assert subset.reader.unique().shape[0] == 1, "More than one reader in subset"
        reader_id = subset["reader"].iloc[0]
        reader_idx = reader_to_idx[reader_id]
        reader_emb = torch.zeros(
            result.shape[0], len(reader_to_idx), dtype=torch.float32
        )
        reader_emb[:, reader_idx] = 1

        position_info = torch.tensor(
            data=result[["x", "y"]].values, dtype=torch.float32
        )
        reader_embs = torch.concatenate([position_info, reader_emb], axis=1)
        if int(char_level_surp) + int(word_level_surp) + int(durations) == 3:
            raise ValueError(
                "Cannot use all three features at once. Please choose one or two."
            )

        if length and freq and word_level_surp and char_level_surp:
            reader_embs_char_word_len_freq = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[
                            [
                                "char_level_surp",
                                "word_level_surprisal",
                                "len",
                                "freq",
                                "saccade_intervals",
                            ]
                        ].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_char_word_len_freq
        if length and freq and word_level_surp:

            reader_embs_len_freq_word_surp = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[
                            ["word_level_surprisal", "len", "freq", "saccade_intervals"]
                        ].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_len_freq_word_surp

        if length and freq and char_level_surp:

            reader_embs_len_freq_char_surp = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[
                            ["char_level_surp", "len", "freq", "saccade_intervals"]
                        ].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_len_freq_char_surp
        if length and freq:
            reader_embs_len_freq = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[["len", "freq", "saccade_intervals"]].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_len_freq

        if durations:
            reader_embs_dur = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[["norm_dur", "saccade_intervals"]].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_dur

        if char_level_surp:

            reader_embs_char_level_surp = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[["char_level_surp", "saccade_intervals"]].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_char_level_surp
        if word_level_surp:

            reader_embs_word_level_surp = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[["word_level_surprisal", "saccade_intervals"]].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_word_level_surp
        if freq:

            reader_embs_freq = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[["freq", "saccade_intervals"]].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_freq
        if length:
            reader_embs_len = torch.concatenate(
                [
                    reader_embs,
                    torch.tensor(
                        subset[["len", "saccade_intervals"]].values,
                        dtype=torch.float32,
                    ),
                ],
                axis=1,
            )
            return reader_embs_len

        return reader_embs
    return torch.tensor(result.values, dtype=torch.float32)


# ********************************#
# * DUR MODEL PREDICTORS         #
# ********************************#


def dur_model_reader_len_freq_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(
        subset, char_df, reader_to_idx, len_conv=True, freq_conv=True
    )


def dur_model_reader_char_len_freq_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(
        subset, char_df, reader_to_idx, len_conv=True, freq_conv=True, char_conv=True
    )


def dur_model_reader_word_len_freq_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(
        subset, char_df, reader_to_idx, len_conv=True, freq_conv=True, word_conv=True
    )


def dur_model_reader_len_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(subset, char_df, reader_to_idx, len_conv=True)


def dur_model_reader_freq_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(subset, char_df, reader_to_idx, freq_conv=True)


def dur_model_reader_dur_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(subset, char_df, reader_to_idx, dur_conv=True)


def dur_model_reader_char_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(subset, char_df, reader_to_idx, char_conv=True)


def dur_model_reader_word_conv_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(subset, char_df, reader_to_idx, word_conv=True)


def dur_model_reader_char_word_len_freq_features(subset, char_df, reader_to_idx):
    return dur_model_reader_features(
        subset,
        char_df,
        reader_to_idx,
        word_conv=True,
        char_conv=True,
        len_conv=True,
        freq_conv=True,
    )


def dur_model_baseline(subset, char_df, reader_to_idx):
    return torch.empty(subset.shape[0], 0, dtype=torch.float32)


def dur_model_reader_features(
    subset,
    char_df,
    reader_to_idx,
    dur_conv=False,
    char_conv=False,
    word_conv=False,
    len_conv=False,
    freq_conv=False,
):
    # GET_READER
    reader_id = subset["reader"].iloc[0]
    reader_idx = reader_to_idx[reader_id]
    reader_emb = torch.zeros(subset.shape[0], len(reader_to_idx), dtype=torch.float32)
    reader_emb[:, reader_idx] = 1

    if (
        (word_conv == False)
        and (char_conv == False)
        and (len_conv == False)
        and (freq_conv == False)
        and (dur_conv == False)
    ):
        return reader_emb

    subset = subset.copy()

    dur_tensor = torch.tensor(subset[["norm_dur", "start"]].values, dtype=torch.float32)
    char_tensor = torch.tensor(
        subset[["char_level_surp", "start", "norm_dur"]].values, dtype=torch.float32
    )

    word_tensor = torch.tensor(
        subset[["word_level_surprisal", "start", "norm_dur"]].values,
        dtype=torch.float32,
    )

    len_tensor = torch.tensor(
        subset[["len", "start", "norm_dur"]].values, dtype=torch.float32
    )

    freq_tensor = torch.tensor(
        subset[["freq", "start", "norm_dur"]].values, dtype=torch.float32
    )

    len_freq_tensor = torch.tensor(
        subset[["len", "freq", "start", "norm_dur"]].values, dtype=torch.float32
    )

    char_len_freq_surp_tensor = torch.tensor(
        subset[["char_level_surp", "len", "freq", "start", "norm_dur"]].values,
        dtype=torch.float32,
    )

    word_len_freq_surp_tensor = torch.tensor(
        data=subset[
            ["word_level_surprisal", "len", "freq", "start", "norm_dur"]
        ].values,
        dtype=torch.float32,
    )

    char_word_len_freq_tensor = torch.tensor(
        data=subset[
            [
                "char_level_surp",
                "word_level_surprisal",
                "len",
                "freq",
                "start",
                "norm_dur",
            ]
        ].values,
        dtype=torch.float32,
    )

    if freq_conv and len_conv and char_conv and word_conv:
        result_emb = torch.concatenate([reader_emb, char_word_len_freq_tensor], axis=1)
        return result_emb

    if freq_conv and len_conv and char_conv:
        result_emb = torch.concatenate([reader_emb, char_len_freq_surp_tensor], axis=1)
        return result_emb
    if freq_conv and len_conv and word_conv:
        result_emb = torch.concatenate([reader_emb, word_len_freq_surp_tensor], axis=1)
        return result_emb
    if freq_conv and len_conv:
        result_emb = torch.concatenate([reader_emb, len_freq_tensor], axis=1)
        return result_emb

    if dur_conv:
        result_emb = torch.concatenate([reader_emb, dur_tensor], axis=1)
        return result_emb

    if char_conv:
        result_emb = torch.concatenate([reader_emb, char_tensor], axis=1)
        return result_emb

    if word_conv:
        result_emb = torch.concatenate([reader_emb, word_tensor], axis=1)
        return result_emb

    if len_conv:
        result_emb = torch.concatenate([reader_emb, len_tensor], axis=1)
        return result_emb

    if freq_conv:
        result_emb = torch.concatenate([reader_emb, freq_tensor], axis=1)
        return result_emb

    raise ValueError(
        "No valid feature combination provided. Please check the parameters."
    )
