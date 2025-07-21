import pandas as pd
import pandas as pd


def get_filtered_scanpath(mc_ds: pd.DataFrame):

    # Initialize an empty list to store aggregated subsets
    aggregated_data = []

    # Iterate through all unique combinations of text and reader
    for text_id in mc_ds["text"].unique():
        for reader_id in mc_ds["reader"].unique():
            # Filter the subset for the current text and reader
            subset = mc_ds[(mc_ds["text"] == text_id) & (mc_ds["reader"] == reader_id)]
            subset = subset[
                (subset["ia_word"].notna())
                & (subset["ia_word"] != " ")
                & (subset["ia_word"] != "lb")
            ]  # Keep only fixations associated to words

            subset = subset.sort_values(by="fixid")  # Ensure ascending order by fixid

            # Is the new fixation on the same word? If so -> we aggregate
            is_new_group = (subset["ia_word"] != subset["ia_word"].shift()) | (
                subset["fixid"].diff() != 1
            )
            #  It's true if the previous fixation is on a different word.
            # This is used to identify when a new word starts in the sequence. (it gets a new index)
            #
            # )

            grouped = subset.groupby(is_new_group.cumsum())
            #   Every True increments the running total, so runs of False
            #   (i.e., consecutive identical words) share the same ID.

            # Aggregate the columns as specified
            for _, group in grouped:
                aggregated_row = {
                    "fixid": tuple(group["fixid"]),
                    "ia_word": group["ia_word"].iloc[0],  # Take the first value
                    "word_bbox_x1": group["word_bbox_x1"].iloc[0],
                    "word_bbox_x2": group["word_bbox_x2"].iloc[0],
                    "word_bbox_y1": group["word_bbox_y1"].iloc[0],
                    "word_bbox_y2": group["word_bbox_y2"].iloc[0],
                    "word_level_surprisal": group["word_level_surprisal"].mean(),
                    "char_level_surp": group["char_level_surp"].mean(),
                    "dur": group["dur"].sum(),  # Sum the durations
                    "ianum_word": group["ianum_word"].iloc[0],
                    "text": text_id,
                    "reader": reader_id,
                }
                aggregated_data.append(aggregated_row)

    # Create a new DataFrame from the aggregated data
    scanpath_filtered_ds = pd.DataFrame(aggregated_data)
    return scanpath_filtered_ds


def get_text_dataset_word_index(texts_df):

    def check_reading_order_using_line(df):
        """
        Check if the DataFrame for a given text is in reading order using the 'line' column.
        Reading order is defined as:
        1. Lower 'line' means an earlier line (top-to-bottom)
        2. Within the same line, order by 'center_x' (left-to-right)

        Returns:
            expected_order (list): The list of indices after sorting.
            df_ordered (DataFrame): The DataFrame sorted in reading order.
        """
        df_ordered = df.sort_values(by=["line", "center_x"], kind="mergesort")
        return df_ordered.index.tolist(), df_ordered

    import pandas as pd

    # List to store the word DataFrame for each text
    word_aggregated_data = []

    # Iterate over each text using texts_df (assumed to have a "text_id" and "ia_word" column)
    for text_id, group in texts_df.groupby("text_id"):
        # Ensure the group is ordered in reading order using the provided function
        _, ordered_df = check_reading_order_using_line(group)

        # Initialize filtered lists for words and bounding boxes
        filtered_words = []
        filtered_bbox_x1 = []
        filtered_bbox_y1 = []
        filtered_bbox_x2 = []
        filtered_bbox_y2 = []

        prev_word = None
        # Iterate over the rows of ordered_df
        for idx, row in ordered_df.iterrows():
            # Skip if word is NaN, 'lb' (line break), or blank after stripping
            if pd.isna(row["ia_word"]):
                continue
            word_str = str(row["ia_word"]).strip()
            if word_str.lower() == "lb" or word_str == "":
                continue
            # Only add if the word is different from the previous one (consecutive duplicate check)
            if word_str != prev_word:
                filtered_words.append(word_str)
                filtered_bbox_x1.append(row["word_bbox_x1"])
                filtered_bbox_y1.append(row["word_bbox_y1"])
                filtered_bbox_x2.append(row["word_bbox_x2"])
                filtered_bbox_y2.append(row["word_bbox_y2"])
                prev_word = word_str

        # Build a DataFrame for this text adding a word_index for ordering
        df_text_words = pd.DataFrame(
            {
                "text_id": text_id,
                "ia_word": filtered_words,
                "word_index": range(1, len(filtered_words) + 1),
                "word_bbox_x1": filtered_bbox_x1,
                "word_bbox_y1": filtered_bbox_y1,
                "word_bbox_x2": filtered_bbox_x2,
                "word_bbox_y2": filtered_bbox_y2,
            }
        )
        word_aggregated_data.append(df_text_words)

    # Concatenate all individual text DataFrames into one
    word_ordered_df = pd.concat(word_aggregated_data, ignore_index=True)
    return word_ordered_df


def gaze_duration_dataset(scanpath_filtered_ds):
    """
    Create a dataset with gaze durations for each word in the scanpath.
    """

    mc_dsw_conflated_unique = scanpath_filtered_ds.groupby(
        ["text", "reader"], group_keys=False
    ).apply(lambda grp: grp.drop_duplicates(subset="ianum_word", keep="first"))
    return mc_dsw_conflated_unique


def get_first_fixation_dataset(mc_ds, word_ordered_df):
    mc_ds_fltr = mc_ds[
        (mc_ds["ia_word"].isna() == False)
        & (mc_ds["ia_word"] != " ")
        & (mc_ds["ia_word"] != "lb")
    ]

    left_keys = [
        "text",
        "word_bbox_x1",
        "word_bbox_x2",
        "word_bbox_y1",
        "word_bbox_y2",
        "ia_word",
    ]

    right_keys = [
        "text_id",
        "word_bbox_x1",
        "word_bbox_x2",
        "word_bbox_y1",
        "word_bbox_y2",
        "ia_word",
    ]

    mc_dsw = pd.merge(
        mc_ds_fltr,
        word_ordered_df,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        suffixes=("", "_y"),
    )
    mc_dsw_first_fixation = mc_dsw.groupby(["text", "reader"], group_keys=False).apply(
        lambda grp: grp.drop_duplicates(subset="word_index", keep="first")
    )
    return mc_dsw_first_fixation


def get_total_duration_dataset(mc_ds, word_ordered_df):
    mc_ds_fltr = mc_ds[
        (mc_ds["ia_word"].isna() == False)
        & (mc_ds["ia_word"] != " ")
        & (mc_ds["ia_word"] != "lb")
    ]

    left_keys = [
        "text",
        "word_bbox_x1",
        "word_bbox_x2",
        "word_bbox_y1",
        "word_bbox_y2",
        "ia_word",
    ]

    right_keys = [
        "text_id",
        "word_bbox_x1",
        "word_bbox_x2",
        "word_bbox_y1",
        "word_bbox_y2",
        "ia_word",
    ]

    ### Total Duration
    mc_dsw = pd.merge(
        mc_ds_fltr,
        word_ordered_df,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        suffixes=("", "_y"),
    )

    total_gaze_df = (
        mc_dsw.groupby(["text", "reader", "ia_word", "word_index"], group_keys=False)
        .agg(
            {
                "fixid": lambda x: tuple(x),  # Concatenate the fixid values
                "word_level_surprisal": "mean",
                "char_level_surp": "mean",
                "dur": "sum",
                "word_bbox_x1": "first",
                "word_bbox_y1": "first",
                "word_bbox_x2": "first",
                "word_bbox_y2": "first",
            }
        )
        .reset_index()
    )

    return total_gaze_df


def add_past_marks(
    gd_ds_augm: pd.DataFrame,
    agg_ds: pd.DataFrame,
    k: int = 5,
    *,
    word_col_gd: str = "ia_word",
    word_col_agg: str = "ia",
    trial_col_gd: str = "text",
    trial_col_agg: str = "trialid",
    idx_gd: str = "ianum_word",
    idx_agg: str = "ianum",
    past_cols=("freq", "surp", "len", "dur"),
) -> pd.DataFrame:
    # Harmonize column names for merging
    ag = agg_ds.copy().rename(
        columns={
            trial_col_agg: trial_col_gd,
            idx_agg: idx_gd,
            word_col_agg: word_col_gd,
        }
    )

    # Sort the aggregate dataframe by trial and word index
    ag = ag.sort_values([trial_col_gd, idx_gd])

    # Create past shifted columns within each trial
    for var in past_cols:
        for i in range(1, k + 1):
            if k == 1:
                ag[f"prev_{var}"] = ag.groupby(by=trial_col_gd)[var].shift(i)
            if k > 1:
                ag[f"prev_{k}_{var}"] = ag.groupby(by=trial_col_gd)[var].shift(i)

    # Keep only necessary columns for merging
    past_columns = [f"{v}_past_{i}" for v in past_cols for i in range(1, k + 1)]
    merge_columns = [trial_col_gd, word_col_gd, idx_gd]
    ag_small = ag[merge_columns + past_columns]

    # Merge the augmented past columns onto gd_ds_augm
    gd_augm = gd_ds_augm.merge(ag_small, on=merge_columns, how="left")

    return gd_augm
