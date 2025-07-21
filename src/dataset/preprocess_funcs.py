import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pdb
import torch
import os
from PIL import Image, ImageDraw
from src.dataset.dataset_visuals import batch_plot_character_surp, batch_plot_word_surp
from src.dataset.feature_names import (
    BACKWARD_FIXATION_SUBSET_NAME,
    BACKWARD_PRECISE_FIXATION_SUBSET_NAME,
    FIRST_WORD_FIXATION_SUBSET_NAME,
    FORWARD_FIXATION_SUBSET_NAME,
    FORWARD_PRECISE_FIXATION_SUBSET_NAME,
    NEW_LINE_FIXATION_SUBSET_NAME,
    SAME_WORD_FIXATION_SUBSET_NAME,
)
from src.paths import DATA_DIR


def add_subset_flags(meco_df, texts_df):

    def is_backward_fixation(current, previous):
        is_curr_fixations_to_the_left = current["x"] - previous["x"] < 0
        is_curr_fixation_same_or_lower_line = current["line"] <= previous["line"]
        return is_curr_fixations_to_the_left & is_curr_fixation_same_or_lower_line

    def is_backward_fixation_previous_word_same_line(current, previous):
        is_curr_fixations_to_the_left = current["x"] - previous["x"] < 0
        is_curr_fixation_same_line = current["line"] == previous["line"]
        is_curr_fixation_a_word = current["ianum_word"] != -1
        is_previous_fixation_a_word = previous["ianum_word"] != -1
        they_are_different_words = current["ianum_word"] != previous["ianum_word"]
        return (
            is_curr_fixations_to_the_left
            & is_curr_fixation_same_line
            & is_curr_fixation_a_word
            & is_previous_fixation_a_word
            & they_are_different_words
        )

    def is_forward_fixation_same_line_next_word(current, previous):
        is_curr_fixations_to_the_right = current["x"] - previous["x"] > 0
        is_curr_fixation_same_line = current["line"] == previous["line"]
        is_curr_fixation_a_word = current["ianum_word"] != -1
        is_previous_fixation_a_word = previous["ianum_word"] != -1
        they_are_different_words = current["ianum_word"] != previous["ianum_word"]
        return (
            is_curr_fixations_to_the_right
            & is_curr_fixation_same_line
            & is_curr_fixation_a_word
            & is_previous_fixation_a_word
            & they_are_different_words
        )

    def is_forward_fixation(current, previous):
        is_curr_fixations_to_the_right = current["x"] - previous["x"] > 0
        is_curr_same_line_or_higher = current["line"] >= previous["line"]
        return is_curr_fixations_to_the_right & is_curr_same_line_or_higher

    def is_same_word_fixation(current, previous):
        is_curr_fixation_in_word = (current["x"] >= previous["word_bbox_x1"]) & (
            current["x"] <= previous["word_bbox_x2"]
        )
        is_curr_fixation_same_line = current["line"] == previous["line"]
        is_curr_fixation_same_word = (
            is_curr_fixation_in_word & is_curr_fixation_same_line
        )
        return is_curr_fixation_same_word

    def is_first_word_fixation(df, current):
        idx = current.name
        df = df.iloc[:idx]

        if pd.isna(current["ianum_word"]):
            return 0
        if df.shape[0] == 0:
            return 1
        is_same_word = df["ianum_word"] == current["ianum_word"]
        return int(not is_same_word.any())

    def is_new_line_fixation(current, previous):
        return current["line"] > previous["line"]

    def compute_flag(row, df, comp_func):
        idx = row.name  # row.name gives the index of the row in the DataFrame
        if idx == 0:
            # For the first row, there is no previous row.
            return 0
        # Get the previous row from the DataFrame.
        previous_row = df.iloc[idx - 1]
        # Your comparison logic: for example, flag if there's a gap between the previous 'end' and the current 'start'

        return 1 if comp_func(row, previous_row) else 0

    subset_meco_list = []
    for text_id in texts_df["text_id"].unique():
        print(f"Processing text {text_id} for flag computation")
        meco_subset = meco_df[meco_df["text"] == text_id]
        meco_subset.reset_index(drop=True, inplace=True)
        meco_subset[BACKWARD_FIXATION_SUBSET_NAME] = meco_subset.apply(
            lambda x: compute_flag(
                row=x, df=meco_subset, comp_func=is_backward_fixation
            ),
            axis=1,
        )
        meco_subset[FORWARD_FIXATION_SUBSET_NAME] = meco_subset.apply(
            lambda x: compute_flag(x, meco_subset, is_forward_fixation), axis=1
        )
        meco_subset[SAME_WORD_FIXATION_SUBSET_NAME] = meco_subset.apply(
            lambda x: compute_flag(
                row=x, df=meco_subset, comp_func=is_same_word_fixation
            ),
            axis=1,
        )
        meco_subset[FIRST_WORD_FIXATION_SUBSET_NAME] = meco_subset.apply(
            lambda x: is_first_word_fixation(meco_subset, x), axis=1
        )
        meco_subset[NEW_LINE_FIXATION_SUBSET_NAME] = meco_subset.apply(
            lambda x: compute_flag(
                row=x, df=meco_subset, comp_func=is_new_line_fixation
            ),
            axis=1,
        )

        meco_subset[BACKWARD_PRECISE_FIXATION_SUBSET_NAME] = meco_subset.apply(
            lambda x: compute_flag(
                row=x,
                df=meco_subset,
                comp_func=is_backward_fixation_previous_word_same_line,
            ),
            axis=1,
        )
        meco_subset[FORWARD_PRECISE_FIXATION_SUBSET_NAME] = meco_subset.apply(
            lambda x: compute_flag(
                row=x, df=meco_subset, comp_func=is_forward_fixation_same_line_next_word
            ),
            axis=1,
        )

        subset_meco_list.append(meco_subset)
    return pd.concat(subset_meco_list)


def process_bounding_boxes_boundaries(
    text_df, division_factor_space, buffer_left_right=0
):
    """
    Adjust bounding box boundaries for a single text:
    - Uniform y1 and y2 values for all characters in the same line.
    - Add horizontal buffers to x1 and x2 based on neighboring characters.

    Parameters:
        text_df (pd.DataFrame): DataFrame containing bounding box information for a single text.

    Returns:
        pd.DataFrame: Processed DataFrame with updated bounding box boundaries.
    """
    # text_df["line"] = text_df["line_break"].cumsum()
    # Uniform y1 and y2 values for all characters in the same line
    text_df["y1_uniform"] = text_df.groupby("line")["bbox_y1"].transform("min")
    text_df["y2_uniform"] = text_df.groupby("line")["bbox_y2"].transform("max")

    # Apply uniform y1 and y2 values
    text_df["bbox_y1"] = text_df["y1_uniform"]
    text_df["bbox_y2"] = text_df["y2_uniform"]
    text_df.drop(columns=["y1_uniform", "y2_uniform"], inplace=True)

    # Add horizontal buffers to x1 and x2
    text_df["prev_character"] = text_df["character"].shift(1)
    text_df["next_character"] = text_df["character"].shift(-1)

    # If the previous character is NaN, add a buffer to x1
    text_df.loc[text_df["prev_character"].isna(), "bbox_x1"] -= (
        buffer_left_right / division_factor_space
    )

    # If the next character is NaN, add a buffer to x2
    text_df.loc[text_df["next_character"].isna(), "bbox_x2"] += (
        buffer_left_right / division_factor_space
    )

    # Drop temporary columns
    text_df.drop(columns=["prev_character", "next_character"], inplace=True)

    return text_df


def bounding_boxes_buffer_wrapper(texts_df, division_factor_space):
    """
    Apply bounding box processing to all texts in the DataFrame.

    Parameters:
        texts_df (pd.DataFrame): DataFrame containing bounding box information for multiple texts.

    Returns:
        pd.DataFrame: Processed DataFrame with updated bounding box boundaries for all texts.
    """
    processed_texts = []
    for text_id in texts_df["text_id"].unique():
        text_subset = texts_df[texts_df["text_id"] == text_id]
        processed_text_subset = process_bounding_boxes_boundaries(
            text_subset, division_factor_space
        )
        processed_texts.append(processed_text_subset)

    return pd.concat(processed_texts, ignore_index=True)


def plot_character_boxes_overlay(
    text_ds: pd.DataFrame,
    input_file: str,
    output_file: str,
    division_factor_space: float,
) -> str:
    """
    Draws bounding boxes around every character in `text_ds` on top of the image.
    - text_ds must contain ['bbox_x1','bbox_y1','bbox_x2','bbox_y2','text_id'].
    - coords in text_ds are in normalized units; we rescale by division_factor_space.
    """
    # subset to single text_id
    text_id = text_ds["text_id"].unique()
    if len(text_id) != 1:
        raise ValueError(f"Expected a single text_id, got {text_id}")
    df = text_ds.copy()
    # rescale to pixels
    for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        df[col] = df[col] * division_factor_space

    # load image and flatten alpha
    im = Image.open(input_file)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    canvas = Image.new("RGB", im.size, (255, 255, 255))
    canvas.paste(im, (0, 0), im)
    draw = ImageDraw.Draw(canvas)

    # draw every character box
    for _, row in df.iterrows():
        box = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
        draw.rectangle(box, outline="green", width=1)

    # save
    os.makedirs(Path(output_file).parent, exist_ok=True)
    canvas.save(output_file)
    return str(output_file)


def batch_plot_character_boxes(
    text_df: pd.DataFrame, images_dir: Path | str, division_factor_space
):
    """
    For each text_id in text_df, overlay character boxes on the corresponding Item_XX.png.
    - text_df must contain ['text_id','bbox_x1','bbox_y1','bbox_x2','bbox_y2'].
    - images_dir is the folder with Item_01.png … Item_12.png, etc.
    """
    images_dir = Path(images_dir)
    out_dir = images_dir.parent / "texts_en_images_char_boxes"
    out_dir.mkdir(exist_ok=True)

    for tid in sorted(text_df["text_id"].unique()):
        subset = text_df[text_df["text_id"] == tid]
        img_file = images_dir / f"Item_{int(tid):02d}.png"
        if not img_file.exists():
            print(f"⚠️  {img_file} not found, skipping text_id={tid}")
            continue
        out_file = out_dir / f"Item_{int(tid):02d}_char_boxes.png"
        saved = plot_character_boxes_overlay(
            text_ds=subset,
            input_file=str(img_file),
            output_file=str(out_file),
            division_factor_space=division_factor_space,
        )
        print(f"Saved character‐box overlay for text {tid} → {saved}")


def dataset_preprocessing_english_text(
    text_df_path,
    meco_dataset_dir,
    word_level_path,
    characters_surps_path,
    division_factor_space,
    division_factor_time,
    division_factor_durations,
    past_timesteps_duration_baseline_k,
):
    # Check that duration is the same
    meco_df = get_normalized_meco_dataframe(
        "en",
        meco_dataset_dir,
        division_factor_space,
        division_factor_time,
        division_factor_durations,
    )

    texts_df = pd.read_csv(filepath_or_buffer=text_df_path)
    texts_df = get_normalized_texts_dataframe(texts_df, division_factor_space)

    assert (texts_df["bbox_x1"] > texts_df["bbox_x2"]).sum() == 0
    assert (texts_df["bbox_y1"] > texts_df["bbox_y2"]).sum() == 0
    texts_df = bounding_boxes_buffer_wrapper(texts_df, division_factor_space)
    assert (texts_df["bbox_x1"] > texts_df["bbox_x2"]).sum() == 0
    assert (texts_df["bbox_y1"] > texts_df["bbox_y2"]).sum() == 0

    batch_plot_character_boxes(
        text_df=texts_df,
        images_dir=DATA_DIR / "MECO" / "texts_en_images",
        division_factor_space=division_factor_space,
    )

    word_level_text = pd.read_csv(word_level_path)
    mapping = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",  # en dash to hyphen
        "—": "-",  # em dash to hyphen
        "-": "-",
    }

    word_level_text["ia"] = word_level_text["ia"].apply(
        lambda x: mapping[x] if x in mapping else x
    )
    texts_df = assign_word_attributes(texts_df, word_level_text)
    texts_df["is_char_pres"] = texts_df.apply(
        lambda df: df["character"] in df["ia_word"], axis=1
    )
    assert texts_df[texts_df["is_char_pres"] == False].empty
    char_df_surp = pd.read_csv(characters_surps_path)
    char_df_surp["center_x"] = char_df_surp["center_x"] / division_factor_space
    char_df_surp["center_y"] = char_df_surp["center_y"] / division_factor_space
    char_df_surp["char_level_surp"] = char_df_surp["surp"]
    text_merged = pd.merge(
        left=texts_df,
        right=char_df_surp[
            ["character", "center_x", "center_y", "text_id", "char_level_surp"]
        ],
        on=["character", "center_x", "center_y", "text_id"],
        how="left",
    )

    assert text_merged.shape[0] == texts_df.shape[0]
    texts_df = text_merged
    texts_df = pd.merge(
        texts_df,
        word_level_text[["ianum", "trialid", "surp", "freq", "len"]],
        left_on=["ianum_word", "trialid_word"],
        right_on=["ianum", "trialid"],
        how="left",
    )

    texts_df = extend_df_with_word_bboxes(texts_df)

    new_texts_df = []
    for text_id in texts_df["text_id"].unique():
        subset_df = texts_df[texts_df["text_id"] == text_id]
        lines_df = pad_line_boundaries(subset_df)
        lines_df.loc[(lines_df["line"] == 0), "padded_lower"] = 0
        lines_df.loc[(lines_df["line"] == lines_df["line"].max()), "padded_upper"] += 3
        lines_df.rename(
            columns={
                "padded_lower": "vertical_line_start",
                "padded_upper": "vertical_line_end",
            },
            inplace=True,
        )
        subset_df_line_coords = pd.merge(
            subset_df,
            lines_df[["vertical_line_start", "vertical_line_end", "line"]],
            on="line",
            how="left",
        )
        if subset_df_line_coords.shape[0] != subset_df.shape[0]:
            pdb.set_trace()
        new_texts_df.append(subset_df_line_coords)

    texts_df = pd.concat(new_texts_df)

    batch_plot_character_surp(
        texts_df,
        DATA_DIR / "MECO" / "texts_en_images",
        division_factor_space,
        out_suffix="char_level_surp",
    )
    batch_plot_word_surp(
        texts_df, DATA_DIR / "MECO" / "texts_en_images", division_factor_space
    )
    texts_df["char_level_surp"] = texts_df.groupby("text_id")[
        "char_level_surp"
    ].transform(lambda s: s.rolling(window=5, center=True, min_periods=1).sum())
    batch_plot_character_surp(
        texts_df,
        DATA_DIR / "MECO" / "texts_en_images",
        division_factor_space,
        out_suffix="char_level_surp_sum",
    )

    meco_df.drop(
        ["letter", "letternum", "right_bounded_time", "cumsum_dur_shift"],
        axis=1,
        inplace=True,
    )

    meco_df = assign_fixations_to_words(meco_df, texts_df)
    meco_df = meco_df[
        [
            "reader",
            "text",
            "fixid",
            "x",
            "y",
            "dur",
            "start",
            "saccade_intervals",
            "character",
            "ia_word",
            "ianum_word",
            "line",
            "char_level_surp",
            "surp",
            "bbox_x1",
            "bbox_x2",
            "bbox_y1",
            "bbox_y2",
            "word_bbox_x1",
            "word_bbox_x2",
            "word_bbox_y1",
            "word_bbox_y2",
            "language",
            "len",
            "freq",
        ]
    ]

    meco_df.loc[meco_df["ianum_word"] == -1, "ia_word"] = np.nan
    meco_df.loc[meco_df["ianum_word"] == -1, "ianum_word"] = np.nan

    subs = meco_df[meco_df["character"] != "lb"]
    assert subs[(subs["char_level_surp"].isna())][
        (subs["character"].isna() == False)
    ].empty

    meco_df.rename(columns={"surp": "word_level_surprisal"}, inplace=True)
    meco_df["word_len"] = meco_df["ia_word"].apply(
        lambda x: len(x) if isinstance(x, str) else np.nan
    )
    # meco_df = wrapper_longitudinal_processing(
    #    meco_df=meco_df, K=past_timesteps_duration_baseline_k
    # )

    return meco_df, texts_df


def normalize_letter(letter):
    """
    Normalize typographic variants to a canonical form.

    For example, this maps:
      - Left and right double quotes (“ and ”) to the standard '"'
      - Left and right single quotes (‘ and ’) to the standard "'"
      - En dash (–) and em dash (—) to the hyphen-minus '-'

    Extend this mapping as needed.
    """
    mapping = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",  # en dash to hyphen
        "—": "-",  # em dash to hyphen
    }
    return mapping.get(letter, letter)


def extend_df_with_word_bboxes(df):
    """
    Given a pandas dataframe with columns including:
      ['character', 'center_x', 'center_y', 'bbox_x1', 'bbox_y2', 'bbox_x2',
       'bbox_y1', 'line_break', 'idx', 'text_id', 'language', 'areas', 'line',
       'line_avg_center_x', 'line_avg_center_y', 'c_value', 'char_idx',
       'is_capitalized', 'ia_word', 'ianum_word', 'trialid_word'],
    this function computes, for each word (identified by the tuple (text_id, ianum_word)
    with ianum_word != -1), a bounding box that encloses all its characters.

    The character bounding box is assumed to be given by:
      - bbox_x1: left coordinate
      - bbox_y1: top coordinate
      - bbox_x2: right coordinate
      - bbox_y2: bottom coordinate
    (Even though the dataframe column order might be different, the names are used here.)

    For any valid word (ianum_word != -1), if any character has a NaN in any of the bounding box
    columns ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'), the function triggers pdb.set_trace() so you can debug.

    The computed word bounding boxes are then merged back into the original dataframe as new columns:
      'word_bbox_x1', 'word_bbox_y1', 'word_bbox_x2', 'word_bbox_y2'

    For rows where ianum_word == -1 (i.e. not a valid word), these new columns will be NaN.

    Returns:
      A new dataframe with the same rows as the input but with four additional columns for the word bounding box.
    """
    # Make a copy so that we don't modify the original dataframe.
    df_out = df.copy()

    # Define the names of the bounding box columns.
    bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

    # Filter for rows corresponding to valid words.
    valid_df = df_out[df_out["ianum_word"] != -1]

    # For each valid word (grouped by text_id and ianum_word), compute the bounding box.
    word_bbox_list = []
    for (text_id, ianum_word), group in valid_df.groupby(["text_id", "ianum_word"]):
        # If any of the bounding box columns contain NaN for any character, drop into the debugger.
        if group[bbox_cols].isna().any().any():
            pdb.set_trace()  # Debug: Found a NaN in one of the bbox columns for a valid word.

        # Compute the word bounding box.
        #   Left (x1): minimum of all bbox_x1 values.
        #   Top (y1): minimum of all bbox_y1 values.
        #   Right (x2): maximum of all bbox_x2 values.
        #   Bottom (y2): maximum of all bbox_y2 values.
        word_bbox = {
            "text_id": text_id,
            "ianum_word": ianum_word,
            "word_bbox_x1": group["bbox_x1"].min(),
            "word_bbox_y1": group["bbox_y1"].min(),
            "word_bbox_x2": group["bbox_x2"].max(),
            "word_bbox_y2": group["bbox_y2"].max(),
        }
        word_bbox_list.append(word_bbox)

    # Build a DataFrame from the computed word bounding boxes.
    word_bbox_df = pd.DataFrame(word_bbox_list)

    # Merge the computed word bounding boxes back into the original dataframe.
    # For rows corresponding to invalid words (ianum_word == -1), the merge will yield NaN.
    df_out = df_out.merge(word_bbox_df, on=["text_id", "ianum_word"], how="left")

    if (
        df_out[["character", "ia_word"]]
        .apply(lambda df: df["character"] not in df["ia_word"], axis=1)
        .sum()
        > 0
    ):
        df["consistency_check"] = df_out[["character", "ia_word"]].apply(
            lambda df: df["character"] not in df["ia_word"], axis=1
        )
        pdb.set_trace()

    return df_out


def assign_fixations_to_words(meco_df, word_df):
    """
    For each fixation in meco_df, assign the word in word_df whose bounding box contains the fixation's (x, y).

    Parameters:
      meco_df: pandas.DataFrame
          DataFrame of fixations. Must contain at least the columns:
            - 'x': the horizontal coordinate of the fixation.
            - 'y': the vertical coordinate of the fixation.
            - 'text_id': the id of the text that the fixation belongs to.

      word_df: pandas.DataFrame
          DataFrame of words. Must contain at least the columns:
            - 'text_id': text identifier.
            - 'ianum_word': a number identifying the word (valid words have ianum_word != -1).
            - 'word_bbox_x1': left x coordinate of the word's bounding box.
            - 'word_bbox_y1': top y coordinate of the word's bounding box.
            - 'word_bbox_x2': right x coordinate of the word's bounding box.
            - 'word_bbox_y2': bottom y coordinate of the word's bounding box.
          Optionally, it may include additional metadata (for example, 'line', 'ia_word').

    For each fixation, the function finds the corresponding word such that:
      word_bbox_x1 <= fixation x <= word_bbox_x2, and
      word_bbox_y1 <= fixation y <= word_bbox_y2.

    If no word bounding box contains the fixation, the new columns are left as NaN.
    If more than one word in the same text qualifies, the function calls pdb.set_trace() for debugging.

    The function returns an augmented copy of meco_df with the following new columns:
      - 'line': the line number (if available from word_df)
      - 'ia_word': the word identifier (if available)
      - 'ianum_word': the numeric word id from word_df
      - 'word_bbox_x1', 'word_bbox_y1', 'word_bbox_x2', 'word_bbox_y2': the bounding box of the word.
    """

    # Work on a copy so we don't modify the original meco_df
    meco_df = meco_df.copy()

    # Create new columns in meco_df; default values are NaN.
    new_columns = [
        "line",
        "ia_word",
        "ianum_word",
        "word_bbox_x1",
        "word_bbox_y1",
        "word_bbox_x2",
        "word_bbox_y2",
    ]
    for col in new_columns:
        meco_df[col] = np.nan

    # Define a function that, for a single fixation row, looks for the word in word_df that contains it.
    def assign_char_to_fixation(row):
        tid = row["text"]
        x = row["x"]
        y = row["y"]
        # Select words for the current text.
        words_in_text = word_df[word_df["text_id"] == tid]
        # Create a boolean mask for words where the fixation falls inside the word's bounding box.
        in_box = (
            (words_in_text["bbox_x1"] <= x)
            & (x <= words_in_text["bbox_x2"])
            & (words_in_text["bbox_y1"] <= y)
            & (y <= words_in_text["bbox_y2"])
        )
        matching_character = words_in_text[in_box]
        if len(matching_character["ia_word"].unique()) > 1:
            matching_character = matching_character[
                matching_character["character"] != " "
            ]

            if len(matching_character["ia_word"].unique()) > 1:
                # More than one word found for the fixation, but they are not whitespace.
                # This is unexpected and should be investigated.
                pdb.set_trace()

            # More than one word found for the fixation: something unexpected happened.
            # raise ValueError(
            #    f"Multiple words found for fixation at ({x}, {y}) in text {tid}. "
            #    "This should not happen. Please check the data."
            # )
        if len(matching_character["ia_word"].unique()) == 1:
            # Exactly one matching word found.
            match = matching_character.iloc[0]

            # If word_df contains a column 'line' and 'ia_word', copy them; otherwise, these will remain NaN.
            row["ia_word"] = match.get("ia_word", np.nan)
            row["ianum_word"] = match.get("ianum_word", np.nan)
            row["word_bbox_x1"] = match.get("word_bbox_x1", np.nan)
            row["word_bbox_y1"] = match.get("word_bbox_y1", np.nan)
            row["word_bbox_x2"] = match.get("word_bbox_x2", np.nan)
            row["word_bbox_y2"] = match.get("word_bbox_y2", np.nan)
            row["bbox_x1"] = match.get("bbox_x1", np.nan)
            row["bbox_y1"] = match.get("bbox_y1", np.nan)
            row["bbox_x2"] = match.get("bbox_x2", np.nan)
            row["bbox_y2"] = match.get("bbox_y2", np.nan)
            row["surp"] = match.get("surp", np.nan)
            row["char_level_surp"] = match.get("char_level_surp", np.nan)
            row["character"] = match.get("character", np.nan)
            row["freq"] = match.get("freq", np.nan)
            row["len"] = match.get("len", np.nan)

        line = words_in_text[
            (row["y"] >= words_in_text["vertical_line_start"])
            & (row["y"] <= words_in_text["vertical_line_end"])
        ]["line"].unique()
        if len(line) > 1:
            pdb.set_trace()
        row["line"] = line[0] if len(line) == 1 else np.nan

        # else: if no matching word is found, leave the new columns as NaN.
        return row

    # Apply the assignment function row by row.
    meco_df = meco_df.apply(assign_char_to_fixation, axis=1)

    return meco_df


def assign_word_attributes(char_df, word_df):
    """
    For every row in char_df (each a character) assign the word-level info
    (ia, ianum, trialid) from word_df by aligning the characters with the words.

    The function expects:
      - In char_df:
         * 'text_id': identifies the text.
         *  'character': A column with the character ).
         * 'idx': the position of the character.
      - In word_df:
         * 'ia': the word (string).
         * 'ianum': the word's index in the text.
         * 'trialid': the identifier matching the text_id in char_df.

    For whitespace and line breaks (assumed to be tokens in whitespace_tokens), placeholder
    values are assigned.

    If a mismatch occurs where the normalized forms of the characters differ,
    pdb.set_trace() is called so you can inspect the issue.
    """
    # Tokens considered as whitespace or non-word (adjust as needed)
    whitespace_tokens = {" ", "lb"}

    # Work on a copy to avoid modifying the original dataframe.
    char_df = char_df.copy()

    # Create new columns to hold the word-level information.
    char_df["ia_word"] = np.nan
    char_df["ianum_word"] = np.nan
    char_df["trialid_word"] = np.nan

    # Process each text separately (matching char_df['text_id'] with word_df['trialid'])
    for text in char_df["text_id"].unique():
        # Subset and sort the characters and words for this text.
        char_mask = char_df["text_id"] == text
        char_sub = char_df[char_mask].sort_values("idx")
        word_sub = word_df[word_df["trialid"] == text].sort_values("ianum")

        # Get the list of indices and characters so we can update the original df.
        char_indices = char_sub.index.tolist()
        char_list = char_sub["character"].tolist()
        c_ptr = 0  # pointer into the char_list

        # For each word in the text, assign its info to the matching characters.
        for _, word_row in word_sub.iterrows():
            word = word_row["ia"]
            # For each letter in the word:
            for letter in word:
                # Skip over any whitespace tokens and assign placeholders.
                while c_ptr < len(char_list) and char_list[c_ptr] in whitespace_tokens:
                    char_df.loc[char_indices[c_ptr], "ia_word"] = char_list[
                        c_ptr
                    ]  # using the token as placeholder
                    char_df.loc[char_indices[c_ptr], "ianum_word"] = (
                        -1
                    )  # -1 indicates non-word character
                    char_df.loc[char_indices[c_ptr], "trialid_word"] = text
                    c_ptr += 1

                # If we unexpectedly run out of characters, enter the debugger.
                if c_ptr >= len(char_list):
                    print(
                        "Unexpected: ran out of characters while processing word:", word
                    )
                    pdb.set_trace()
                    break

                # Compare the normalized versions of the expected and actual characters.
                actual_char = char_list[c_ptr]
                if normalize_letter(actual_char) != normalize_letter(letter):
                    print(
                        f"Mismatch detected in text '{text}': expected letter {letter!r} but got {actual_char!r} at character index {c_ptr}"
                    )
                    pdb.set_trace()

                # Assign the word-level info to the current character.
                char_df.loc[char_indices[c_ptr], "ia_word"] = word_row["ia"]
                char_df.loc[char_indices[c_ptr], "ianum_word"] = word_row["ianum"]
                char_df.loc[char_indices[c_ptr], "trialid_word"] = word_row["trialid"]
                c_ptr += 1

        # After processing all words, assign placeholder values to any remaining characters.
        while c_ptr < len(char_list):
            char_df.loc[char_indices[c_ptr], "ia_word"] = char_list[c_ptr]
            char_df.loc[char_indices[c_ptr], "ianum_word"] = -1
            char_df.loc[char_indices[c_ptr], "trialid_word"] = text
            c_ptr += 1

    return char_df


def process_line_breaks(df):
    # Sort to ensure correct order within text groups
    df = df.sort_values(["text_id", "idx"])

    # Create shifted columns for previous values within each text group
    df["prev_bbox_x2"] = df.groupby("text_id")["bbox_x2"].shift(1)
    df["prev_bbox_y1"] = df.groupby("text_id")["bbox_y1"].shift(1)
    df["prev_bbox_y2"] = df.groupby("text_id")["bbox_y2"].shift(1)

    # Identify line breaks and valid previous values
    mask = (df["line_break"] == 1) & df["prev_bbox_x2"].notna()

    # Update coordinates for valid line breaks
    df.loc[mask, "bbox_x1"] = df.loc[mask, "prev_bbox_x2"]
    df.loc[mask, "bbox_x2"] = df.loc[mask, "prev_bbox_x2"] + 18
    df.loc[mask, "bbox_y1"] = df.loc[mask, "prev_bbox_y1"]
    df.loc[mask, "bbox_y2"] = df.loc[mask, "prev_bbox_y2"]

    # Calculate new centroids
    df.loc[mask, "center_x"] = (df.loc[mask, "bbox_x1"] + df.loc[mask, "bbox_x2"]) / 2
    df.loc[mask, "center_y"] = (df.loc[mask, "bbox_y1"] + df.loc[mask, "bbox_y2"]) / 2

    # Add 'lb' character
    df.loc[mask, "character"] = "lb"

    # Cleanup temporary columns
    df = df.drop(columns=["prev_bbox_x2", "prev_bbox_y1", "prev_bbox_y2"])

    return df.reset_index(drop=True)


def get_normalized_texts_dataframe(texts_df, division_factor_space):
    # values of y1 are higher than y2
    texts_df.rename(columns={"bbox_y2": "bbox_y1", "bbox_y1": "bbox_y2"}, inplace=True)
    texts_df = process_line_breaks(texts_df)

    ###############################
    # RESCALE SPATIAL COORDINATES #
    ###############################

    coord_cols = [
        "center_x",
        "center_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    texts_df[coord_cols] = texts_df[coord_cols] / division_factor_space
    texts_df = invert_bbox_for_spaces(texts_df)

    texts_df["areas"] = texts_df.apply(
        lambda df: (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"]),
        axis=1,
    )
    texts_df = assign_lines(texts_df)
    line_avg_positions = compute_line_avg_positions(texts_df)
    texts_df = pd.merge(texts_df, line_avg_positions, on=["text_id", "line"])

    mapping = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",  # en dash to hyphen
        "—": "-",  # em dash to hyphen
        "-": "-",
    }

    texts_df["character"] = texts_df["character"].apply(
        lambda x: mapping[x] if x in mapping else x
    )
    return texts_df


def invert_bbox_for_spaces(dataframe):
    """Inverts the values of bbox_y2 and bbox_y1 for rows where the character is a space."""
    # Identify rows where the character is a space
    space_mask = dataframe["character"] == " "

    # Swap bbox_y2 and bbox_y1 for these rows
    dataframe.loc[space_mask, ["bbox_y1", "bbox_y2"]] = dataframe.loc[
        space_mask, ["bbox_y2", "bbox_y1"]
    ].values

    return dataframe


def compute_line_avg_positions(texts_df):
    """
    Given a DataFrame `texts_df` with a 'line' column and the columns 'center_x' and 'center_y',
    compute the average x and y positions for each line within each text_id.
    """
    # Group by text_id and line, then calculate the mean of center_x and center_y
    line_avg_df = texts_df.groupby(["text_id", "line"], as_index=False).agg(
        line_avg_center_x=("center_x", "mean"), line_avg_center_y=("center_y", "mean")
    )
    return line_avg_df


def assign_lines(texts_df):
    """
    Given a DataFrame with a 'line_break' column, assign a line number
    to each row (character) for each text (grouped by 'text_id').
    We start at line 0 and whenever we encounter a row with line_break==1,
    the next row will get the previous line + 1.
    """
    # Sort by text_id and idx to ensure proper order
    texts_df = texts_df.sort_values(["text_id", "idx"]).copy()

    def assign_line_to_group(group):
        current_line = 0
        line_numbers = []
        for _, row in group.iterrows():
            # Assign the current line number to this row
            line_numbers.append(current_line)
            # If this row indicates a line break, increment for the next row
            if row["line_break"] == 1:
                current_line += 1
        return pd.Series(line_numbers, index=group.index)

    # Apply the function to each group (each text)
    texts_df["line"] = texts_df.groupby("text_id", group_keys=False).apply(
        assign_line_to_group
    )

    return texts_df


def get_normalized_meco_dataframe(
    language,
    meco_path,
    division_factor_space,
    division_factor_time,
    division_factor_durations,
):
    """The meco values are divided by a constant for optimization purposes."""
    meco_df = MACO_from_csv(meco_path, language)
    meco_df["x"] = meco_df["x"] + 0.1
    meco_df["y"] = meco_df["y"] + 0.1
    meco_df["x"] = meco_df["x"] / division_factor_space
    meco_df["y"] = meco_df["y"] / division_factor_space
    meco_df["start"] = meco_df["start"] / division_factor_time
    meco_df["stop"] = meco_df["stop"] / division_factor_time
    meco_df["dur_norm"] = meco_df["dur"] / division_factor_time
    meco_df["cumsum_dur"] = meco_df.groupby(["reader", "text"])["dur_norm"].transform(
        "cumsum"
    )

    # Shift the cumulative sum within each (reader, text) group by 1, filling the first value in each group with 0
    meco_df["cumsum_dur_shift"] = meco_df.groupby(["reader", "text"])[
        "cumsum_dur"
    ].shift(1, fill_value=0)

    meco_df.drop(columns=["cumsum_dur"], inplace=True)
    meco_df.drop(columns=["dur_norm"], inplace=True)
    # Compute the 'new_start' based on the group-specific shifted cumulative durations
    meco_df["saccade_intervals"] = meco_df["start"] - meco_df["cumsum_dur_shift"]
    meco_df["dur"] = meco_df["dur"] / division_factor_durations

    return meco_df


def MACO_from_csv(directory_path, language):
    """
    Reads all CSV files from a specified directory following the 'en_X_Y.csv' pattern,
    combines them into a single DataFrame, and adds 'reader' and 'text' columns
    based on the file name.

    Parameters:
        directory_path (str): The path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: A combined DataFrame with all CSV data and additional metadata.
    """
    # Get a list of all CSV files in the directory matching the pattern
    csv_files = glob.glob(str(Path(directory_path) / f"{language}_*.csv"))

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through each CSV file
    for file in csv_files:
        # Extract reader and text from the filename using the pattern en_X_Y.csv
        base_name = os.path.basename(file)
        parts = base_name.replace("en_", "").replace(".csv", "").split("_")
        reader = int(parts[0])
        text = int(parts[1])

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Add the reader and text columns
        df["reader"] = reader
        df["text"] = text

        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df["language"] = language  # Add a language column for easy filtering later

    return combined_df


def pad_line_boundaries(words_in_text: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame `words_in_text` with columns 'bbox_y1', 'bbox_y2', and 'line',
    first aggregate the original vertical extents (min bbox_y1 and max bbox_y2) per line.
    Then, pad the boundaries between consecutive lines so that the resulting intervals
    cover the entire y-domain continuously.

    Returns a DataFrame with one row per line and the following new columns:
      - padded_lower: The padded (final) lower y-bound for the line.
      - padded_upper: The padded (final) upper y-bound for the line.
      - padded_range: The height of the padded interval (padded_upper - padded_lower).
    """
    # 1. Aggregate the boundaries per line.
    agg_df = (
        words_in_text.groupby("line")
        .agg(bbox_y1_min=("bbox_y1", "min"), bbox_y2_max=("bbox_y2", "max"))
        .reset_index()
    )

    # Ensure the DataFrame is sorted by line.
    agg_df = agg_df.sort_values("line").reset_index(drop=True)

    padded_lower = []
    padded_upper = []

    num_lines = len(agg_df)

    # 2. Compute padded boundaries.
    for i, row in agg_df.iterrows():
        # For the padded lower boundary:
        if i == 0:
            # First line: use its original lower boundary.
            lower = row["bbox_y1_min"]
        else:
            # For subsequent lines, take the midpoint between the previous line's upper
            # and this line's lower.
            prev_upper = agg_df.loc[i - 1, "bbox_y2_max"]
            curr_lower = row["bbox_y1_min"]
            lower = (prev_upper + curr_lower) / 2
        padded_lower.append(lower)

        # For the padded upper boundary:
        if i == num_lines - 1:
            # Last line: use its original upper boundary.
            upper = row["bbox_y2_max"]
        else:
            # For other lines, take the midpoint between this line's original upper and
            # the next line's original lower.
            curr_upper = row["bbox_y2_max"]
            next_lower = agg_df.loc[i + 1, "bbox_y1_min"]
            upper = (curr_upper + next_lower) / 2
        padded_upper.append(upper)

    agg_df["padded_lower"] = padded_lower
    agg_df["padded_upper"] = padded_upper
    agg_df["padded_range"] = agg_df["padded_upper"] - agg_df["padded_lower"]

    return agg_df


#########################################
# 1) Building [T, N_max, 5] from a DataFrame
#########################################
def create_boxes_tensor_from_dataframe(
    df,
    x1_col="bbox_x1",
    x2_col="bbox_x2",
    y1_col="bbox_y1",
    y2_col="bbox_y2",
    c_col="is_capitalized",
    textid_col="text_id",
):
    """
    Reads bounding-box info from a pandas DataFrame for many text_ids,
    and transforms them into a single 3D torch.Tensor of shape [T, N_max, 5].
    The last dimension is [x1_min, x1_max, y1_min, y1_max, c].

    Returns:
        boxes_3d : FloatTensor of shape [T, N_max, 5]
                   (zero-padded for texts with fewer boxes)
        text_ids : list of unique text_ids in sorted order
    """
    text_ids = sorted(df[textid_col].unique())
    boxes_by_text = []
    lengths = []

    for tid in text_ids:
        sub_df = df[df[textid_col] == tid]

        x1_np = sub_df[x1_col].values
        x2_np = sub_df[x2_col].values
        y1_np = sub_df[y1_col].values
        y2_np = sub_df[y2_col].values

        centroid_x = (x1_np + x2_np) / 2
        centroid_y = (y1_np + y2_np) / 2

        c_np = sub_df[c_col].values
        char_idx = sub_df["char_idx"].values
        char_order_idx = sub_df["idx"]

        # shape [N_i, 5]
        data_np = np.stack(
            arrays=[
                x1_np,
                x2_np,
                y1_np,
                y2_np,
                c_np,
                char_idx,
                centroid_x,
                centroid_y,
                char_order_idx,
            ],
            axis=1,
        )
        data_t = torch.from_numpy(data_np).float()  # => shape [N_i, 5]

        boxes_by_text.append(data_t)
        lengths.append(data_t.shape[0])

    T = len(text_ids)
    N_max = max(lengths) if T > 0 else 0

    # tensor of all -1
    boxes_3d = torch.full(size=(T, N_max, 9), fill_value=-1, dtype=torch.float32)
    for i, data_t in enumerate(boxes_by_text):
        n_i = data_t.shape[0]
        boxes_3d[i, :n_i, :] = data_t

    centroids = boxes_3d[:, :, 6:8]
    char_order_id = boxes_3d[:, :, 8]
    boxes_3d = boxes_3d[:, :, :6]

    ############################
    # CREATE ONE HOT ENCODERS  #
    ############################
    char_ids = boxes_3d[:, :, 5].to(torch.int64)
    N = char_ids.max() + 1
    mask = boxes_3d[:, :, 5] == -1

    one_hot = torch.zeros(
        boxes_3d.shape[0], boxes_3d.shape[1], int(N), dtype=torch.float32
    )

    char_ids[mask] = 0
    one_hot_encodings = one_hot.scatter(2, char_ids.unsqueeze(-1), 1)
    one_hot_encodings[mask] = -1

    return (
        boxes_3d[:, :, :-1],
        one_hot_encodings,
        centroids,
        text_ids,
        char_order_id,
    )


def wrapper_longitudinal_processing(meco_df: pd.DataFrame, K: int = 10) -> pd.DataFrame:
    """
    For each (reader, text) subset, sort by fixid and call get_longitudinal_predictors.
    Returns the augmented meco_df with new columns for the last K fixations.
    """
    # Collect augmented subsets here
    augmented_list = []

    readers = meco_df["reader"].unique()
    texts = meco_df["text"].unique()

    for rdr in readers:
        for txt in texts:
            subset = meco_df[(meco_df["reader"] == rdr) & (meco_df["text"] == txt)]
            if subset.empty:
                continue
            subset = subset.sort_values(by="fixid", ascending=True)
            # Add predictor columns
            subset_aug = get_longitudinal_predictors(subset, K=K)
            augmented_list.append(subset_aug)

    # Concatenate all subsets back
    return pd.concat(augmented_list, ignore_index=True)


def get_longitudinal_predictors(df: pd.DataFrame, K: int = 10) -> pd.DataFrame:
    """
    For each fixation, create K columns for the last K fixations of each relevant column.
    The relevant columns are: ['dur', 'char_level_surp', 'word_level_surprisal','word_len'].
    If there's no past fixation i steps behind, fill with -1.
    """
    # Columns of interest
    cols = ["dur", "char_level_surp", "word_level_surprisal", "word_len"]

    # For each row, we find the previous K fixations
    for k in range(1, K + 1):
        for c in cols:
            new_col = f"{c}_prev_{k}"
            df[new_col] = -1  # default
    indices = df.index.to_list()

    for i in range(len(indices)):
        row_idx = indices[i]
        # find up to K previous
        start = max(0, i - K)
        relevant_indices = indices[start:i]
        # reversed order so the last fixation is the first
        relevant_indices = relevant_indices[::-1]

        # fill columns for each step behind
        for step in range(1, K + 1):
            if step <= len(relevant_indices):
                prev_idx = relevant_indices[step - 1]
                for c in cols:
                    df.at[row_idx, f"{c}_prev_{step}"] = df.at[prev_idx, c]

    return df
