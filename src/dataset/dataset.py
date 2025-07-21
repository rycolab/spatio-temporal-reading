import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from consts import (
    ENABLE_DATA_VISUALIZATION,
)
from dataset.feature_names import SUBSETS_MASKS
from paths import (
    CHARACTER_SURPS_PATH,
    IMAGES_MACO_DIR,
    WORDS_SURPS_PATH,
    DATA_DIR,
    MACO_DATASET_DIR,
    TEXTS_DF_PATH,
)
from src.dataset.dataset_visuals import batch_plot_fixations
from src.dataset.preprocess_funcs import *

#################
# DATASET CLASS #
#################


class MecoDataset(Dataset):

    def __init__(
        self,
        mode,
        filtering,
        splitting_procedure,
        feature_func_stpp,
        feature_func_dur,
        division_factor_space,
        division_factor_time,
        division_factor_durations,
        past_timesteps_duration_baseline_k,
        cfg,
        language="en",
    ):
        super().__init__()

        if filtering not in ["filtered", "raw"]:
            raise ValueError("Filtering must be either 'filtered' or 'raw'")

        if splitting_procedure != "random_shuffle":
            raise ValueError("splitting_procedure not supported")

        if language != "en":
            raise ValueError("Only English is supported")

        ##########################
        # LOAD MACO DATAFRAME     #
        ##########################

        cached_dir = DATA_DIR / "dataset_cached"

        meco_cached_filename = f"hp_augmented_meco_{division_factor_space}_{division_factor_time}_{division_factor_durations}_{past_timesteps_duration_baseline_k}.csv"

        texts_cached_filename = f"hp_eng_texts_{division_factor_space}_{division_factor_time}_{division_factor_durations}_{past_timesteps_duration_baseline_k}.csv"
        self.cfg = cfg
        self.meco_df, texts_df = self.get_datasets(
            cached_directory=cached_dir,
            meco_cached_filename=meco_cached_filename,
            texts_cached_filename=texts_cached_filename,
            division_factor_space=division_factor_space,
            division_factor_time=division_factor_time,
            division_factor_durations=division_factor_durations,
            past_timesteps_duration_baseline_k=past_timesteps_duration_baseline_k,
        )

        total_na_values_freq = self.meco_df["freq"].isna().sum()
        self.meco_df["freq"] = -np.log(self.meco_df["freq"] + 1e-9)
        self.meco_df["dur"] = np.log(self.meco_df["dur"] + 1e-9)

        assert total_na_values_freq == self.meco_df["freq"].isna().sum()

        if ENABLE_DATA_VISUALIZATION and mode == "train":
            print(
                "Visualizing scanpaths on the Meco dataset texts. This may take a while."
            )
            batch_plot_fixations(self.meco_df, texts_df, IMAGES_MACO_DIR)

        self.reader_to_idx = {
            reader: idx
            for idx, reader in enumerate(self.meco_df["reader"].sort_values().unique())
        }

        ###############################
        # ENCODE CHARACTERS AS NUMBERS #
        ###############################

        unique_chars = sorted(
            "".join(texts_df["character"].apply(lambda x: x.lower()).unique())
        )
        self.filtering = filtering
        unique_chars = list(set(unique_chars))
        unique_chars.append("lb")
        char_to_idx = {char: i for i, char in enumerate(unique_chars)}
        self.idx_to_char = {i: char for i, char in enumerate(iterable=unique_chars)}

        ###############################
        # CREATE BOXES TENSOR         #
        ###############################

        self.texts_df = texts_df
        self.texts_df["c_value"] = 1
        self.texts_df["char_idx"] = self.texts_df["character"].apply(
            lambda x: char_to_idx[x.lower()]
        )
        self.texts_df["is_capitalized"] = self.texts_df["character"].apply(
            lambda x: x.isupper()
        )
        # check well processing of one hot encoders
        # ids = (self.one_hot == 1).nonzero(as_tuple = False)[:, -1]
        # "".join([idx_to_char[id.item()] for id in ids])

        self.boxes, self.one_hot, self.boxes_centroid, text_ids, self.char_info = (
            create_boxes_tensor_from_dataframe(self.texts_df)
        )

        ###############################
        # CHECK CHARACTER ORDER       #
        ###############################
        for row in self.char_info:
            mask = row != -1  # Identify values before reaching -1
            valid_values = row[mask]  # Extract values before -1
            if not torch.all(
                valid_values[1:] == valid_values[:-1] + 1
            ):  # Check increment condition
                raise ValueError("Invalid character order")

        ###############################
        # CHECK MASK CONSISTENCY      #
        ###############################
        mask_centroid = (self.boxes_centroid == -1).all(axis=2)
        mask_boxes = (self.boxes == -1).all(axis=2)
        mask_one_hot = (self.one_hot == -1).all(axis=2)
        if (mask_centroid != mask_boxes).any() or (mask_centroid != mask_one_hot).any():
            raise ValueError(
                "Inconsistent masks: boxes, one_hot, and centroid masks do not match."
            )

        self.texts_df["x_diff"] = self.texts_df["bbox_x2"] - self.texts_df["bbox_x1"]
        self.texts_df["y_diff"] = self.texts_df["bbox_y2"] - self.texts_df["bbox_y1"]

        ###############################
        # CREATE DATASET SPLIT        #
        ###############################

        # WE FIX A RANDOM SEED FOR THE DATASET, THAT ENSURES THAT THE SPLIT IS CONSISTENT AMONG DIFFERENT DATASET INSTANCES AND RUNS
        self.rnd_seed = 1242
        self.rnd = np.random.RandomState(self.rnd_seed)
        self.feature_func_stpp = feature_func_stpp
        self.feature_func_dur = feature_func_dur

        if self.filtering == "filtered":
            self.meco_df = self.meco_df[self.meco_df["ianum_word"].isna() == False]
            self.meco_df = self.meco_df.sort_values(
                ["text", "reader", "fixid"], ascending=True
            )
            self.meco_df["fixid"] = (
                self.meco_df.groupby(["text", "reader"]).cumcount() + 1
            )

        train_items, valid_items, test_items = self.get_splitting(
            splitting_procedure=splitting_procedure
        )

        self.normalize_predictors = True

        if mode == "train":
            self.items = train_items
        elif mode == "valid":
            self.items = valid_items
        elif mode == "test":
            self.items = test_items
        self.indices = {idx: item for idx, item in enumerate(self.items)}
        dic = {
            **dict.fromkeys(train_items, "train"),
            **dict.fromkeys(valid_items, "valid"),
            **dict.fromkeys(test_items, "test"),
        }
        self.meco_df["split"] = [
            (
                "held_out_session"
                if (text == 3 and reader == 70)
                else dic[(text, reader, fixid - 1)]
            )
            for text, reader, fixid in zip(
                self.meco_df.text, self.meco_df.reader, self.meco_df.fixid
            )
        ]
        self.meco_df.to_csv(
            path_or_buf=DATA_DIR / f"meco_df_{self.filtering}.csv", index=False
        )

        if self.normalize_predictors:
            # Work on a copy to avoid modifying the original
            self.meco_df = self.meco_df.copy()

            # Split into train, valid, test
            train_df = self.meco_df[self.meco_df["split"] == "train"]

            # Columns to normalize and mapping for output names
            predictor_cols = [
                "freq",
                "len",
                "char_level_surp",
                "word_level_surprisal",
                "dur",
            ]
            output_cols = [
                "freq",
                "len",
                "char_level_surp",
                "word_level_surprisal",
                "norm_dur",
            ]

            # Compute training set min and max
            train_min = train_df[predictor_cols].min()
            train_max = train_df[predictor_cols].max()

            # Apply normalization using train stats
            for orig_col, out_col in zip(predictor_cols, output_cols):
                self.meco_df[out_col] = (
                    self.meco_df[orig_col] - train_min[orig_col]
                ) / (train_max[orig_col] - train_min[orig_col])

    def get_readers(self, text="all"):
        if text == "all":
            return self.meco_df["reader"].unique()
        else:
            return self.meco_df[self.meco_df["text"] == text]["reader"].unique()

    def get_texts(self):
        return self.meco_df["text"].unique()

    def get_splitting(self, splitting_procedure):

        texts_readers_df = self.meco_df[["text", "reader"]].value_counts().reset_index()
        assert self.meco_df.shape[0] == texts_readers_df["count"].sum()

        if splitting_procedure == "random_shuffle":
            HOLD_OUT_SESSION = (3, 70)
            self.held_out_reader = tuple((HOLD_OUT_SESSION[0], HOLD_OUT_SESSION[1], 1))
            texts_readers_df = texts_readers_df.query(
                f"not (text == {HOLD_OUT_SESSION[0]} and reader == {HOLD_OUT_SESSION[1]})"
            )

        item_corpus = list(texts_readers_df.itertuples(index=False, name=None))

        if splitting_procedure == "random_shuffle":
            self.rnd.shuffle(item_corpus)
            sample_list = [
                (idx_text, idx_reader, single_count)
                for idx_text, idx_reader, obs_count in item_corpus
                for single_count in range(obs_count)
            ]

            self.rnd.shuffle(sample_list)
            train_items = sample_list[: int(0.8 * len(sample_list))]
            valid_items = sample_list[
                int(0.8 * len(sample_list)) : int(0.9 * len(sample_list))
            ]
            test_items = sample_list[int(0.9 * len(sample_list)) :]
            return train_items, valid_items, test_items

    def __getitem__(self, index):

        # we keep an index for an held out reading process to visualize ad the end
        item = self.indices[index] if index != -1 else self.held_out_reader

        #####################################################
        # RETRIEVE ITEM (TEXT, READER, POINT)               #
        #####################################################

        text = item[0]
        reader = item[1]
        current_observation = item[2]

        #####################################################
        # FILTER SUBSET FROM MACO DATAFRAME                 #
        #####################################################

        subset = self.meco_df[
            (self.meco_df["text"] == text) & (self.meco_df["reader"] == reader)
        ]

        #############################
        # SACCADE TIMES & LOCATIONS #
        #############################

        history_points = torch.tensor(
            subset[["saccade_intervals", "x", "y"]].values, dtype=torch.float32
        )

        ########################################
        # DURATIONS & DURATIONS STARTING TIMES #
        ########################################
        subset = subset.copy()
        durations = torch.tensor(subset["dur"].values, dtype=torch.float32)
        starting_times = torch.tensor(
            data=subset["saccade_intervals"].values, dtype=torch.float32
        )

        duration_onset_real_times = torch.tensor(
            data=subset["start"].values, dtype=torch.float32
        )

        dur_tensor = torch.stack(tensors=(durations, duration_onset_real_times), dim=1)

        ########################################
        #   MODEL INPUT FEATURES               #
        ########################################

        input_features_tensor_stpp = self.feature_func_stpp(
            subset, self.texts_df, self.reader_to_idx
        )
        input_features_tensor_dur = self.feature_func_dur(
            subset, self.texts_df, self.reader_to_idx
        )

        ########################################
        #    CHARACTER LEVEL INFORMATION       #
        ########################################

        box = self.boxes[text - 1]
        centroid = self.boxes_centroid[text - 1]
        one_hot = self.one_hot[text - 1]
        # BOX CONTAINS ONLY  COORDINATES
        box = torch.cat([box[:, :], one_hot], dim=1)
        y1_min_b = box[..., 2].unsqueeze(1)
        y1_max_b = box[..., 3].unsqueeze(1)

        if (history_points == 0.0).any():
            raise ValueError(
                "There are zero values in history_points, which is not allowed."
            )
        if ((y1_max_b - y1_min_b) < 0).any():
            raise ValueError(
                "y1_max_b - y1_min_b is negative, which is not allowed. Check the box coordinates."
            )

        fixations_cond = torch.from_numpy(
            subset[[column for column in SUBSETS_MASKS.keys()]].values
        )

        return_ = (
            input_features_tensor_stpp,
            input_features_tensor_dur,
            history_points,
            history_points[current_observation],
            dur_tensor[current_observation],
            box,
            fixations_cond[current_observation],
        )

        return_visual = (
            input_features_tensor_stpp,
            input_features_tensor_dur,
            history_points,
            dur_tensor,
            box,
            duration_onset_real_times,
        )

        if index != -1:
            return return_
        else:
            return return_visual

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_datasets(
        cached_directory: Path,
        meco_cached_filename: Path,
        texts_cached_filename: Path,
        division_factor_space: float,
        division_factor_time: float,
        division_factor_durations: float,
        past_timesteps_duration_baseline_k: int,
    ):
        meco_cached = cached_directory / meco_cached_filename
        texts_cached = cached_directory / texts_cached_filename
        if cached_directory.exists():
            if meco_cached.exists() and texts_cached.exists():
                print("Loading cached files...")
                meco_df = pd.read_csv(meco_cached)
                texts_df = pd.read_csv(texts_cached)
            else:
                print(
                    "Cache directory exists but one or both files are missing. Running preprocessing..."
                )
                meco_df, texts_df = dataset_preprocessing_english_text(
                    text_df_path=TEXTS_DF_PATH,
                    meco_dataset_dir=MACO_DATASET_DIR,
                    word_level_path=WORDS_SURPS_PATH,
                    characters_surps_path=CHARACTER_SURPS_PATH,
                    division_factor_space=division_factor_space,
                    division_factor_time=division_factor_time,
                    division_factor_durations=division_factor_durations,
                    past_timesteps_duration_baseline_k=past_timesteps_duration_baseline_k,
                )

                meco_df = add_subset_flags(meco_df, texts_df=texts_df)
                meco_df = meco_df.sort_values(["text", "reader", "fixid"]).reset_index(
                    drop=True
                )

                meco_df.to_csv(meco_cached, index=False)
                texts_df.to_csv(texts_cached, index=False)
        else:
            print(
                "Cache directory does not exist. Creating it and running preprocessing..."
            )
            cached_directory.mkdir(parents=True, exist_ok=True)
            meco_df, texts_df = dataset_preprocessing_english_text(
                text_df_path=TEXTS_DF_PATH,
                meco_dataset_dir=MACO_DATASET_DIR,
                word_level_path=WORDS_SURPS_PATH,
                characters_surps_path=CHARACTER_SURPS_PATH,
                division_factor_space=division_factor_space,
                division_factor_time=division_factor_time,
                division_factor_durations=division_factor_durations,
                past_timesteps_duration_baseline_k=past_timesteps_duration_baseline_k,
            )

            meco_df = add_subset_flags(meco_df, texts_df=texts_df)
            meco_df = meco_df.sort_values(["text", "reader", "fixid"]).reset_index(
                drop=True
            )
            meco_df.to_csv(meco_cached, index=False)
            texts_df.to_csv(texts_cached, index=False)
        return meco_df, texts_df
