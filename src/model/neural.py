import torch
import torch.nn as nn
import torch.nn.functional as F


from dataset.feature_names import (
    LOCATION_L,
    LOCATION_R,
    MARK_INDEX_L,
    MARK_INDEX_R,
    READER_EMB_L,
    READER_EMB_L_DUR,
    READER_EMB_R,
    READER_EMB_R_DUR,
)

from src.model.gamma_conv import gamma_convolution
from ignore.gamma_conv2 import GammaConvolution


class MarkedPointProcess(nn.Module):
    def __init__(
        self,
        duration_prediction_func,
        hawkes_predictors_func,
        model_type,
        cfg,
        logger,
    ):
        super(MarkedPointProcess, self).__init__()

        self.inter_reader_mark = True
        logger.info(
            f"Initializing MarkedPointProcess "
            f"hawkes_predictors_func: {hawkes_predictors_func}, model_type: {model_type}"
            f"Interation reader mark: {self.inter_reader_mark}"
        )
        self.duration_prediction_func = duration_prediction_func
        self.hawkes_predictors_func = hawkes_predictors_func
        self.model_type = model_type
        self.cfg = cfg
        self.logger = logger
        self.track_gamma_values = {
            "gamma_alpha": [],
            "gamma_beta": [],
            "gamma_delta": [],
        }

        if self.cfg.missing_value_effects == "linear_term":
            self.missing_value_term_condition = cfg.dataset_filtering == "raw"
        else:
            self.missing_value_term_condition = False
        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            if self.model_type == "saccade":
                self.initialize_parameters_saccades()

            if self.model_type == "duration":
                self.initialize_parameters_duration()

    def forward(self, stpp_input, dur_input, curr_duration_time):

        if self.model_type == "saccade":
            return self.forward_saccades(stpp_input)
        elif self.model_type == "duration":
            return self.forward_duration(dur_input, curr_duration_time)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def forward_saccades(self, stpp_input):

        duration_effect = "duration" in self.hawkes_predictors_func
        char_effect = "char" in self.hawkes_predictors_func
        word_effect = "word" in self.hawkes_predictors_func
        len_effect = "len" in self.hawkes_predictors_func
        freq_effect = "freq" in self.hawkes_predictors_func

        # * -------------------------- *
        # * Input Features Extraction *
        # * -------------------------- *

        past_fixations_location = stpp_input[:, :, LOCATION_L:LOCATION_R]

        past_fixation_mark = None
        if duration_effect or char_effect or word_effect or len_effect or freq_effect:
            past_fixation_mark = stpp_input[:, :, MARK_INDEX_L:MARK_INDEX_R]

        location_output = self.position_shift(past_fixations_location)

        if "reader" in self.hawkes_predictors_func:
            reader_encoders = stpp_input[:, :, READER_EMB_L:READER_EMB_R]
            reader_effect_loc = self.reader_linear_hawks(reader_encoders)
            alpha = self.reader_linear_alpha(reader_encoders)
            beta = self.reader_linear_beta(reader_encoders)
            location_output = location_output + reader_effect_loc
        else:
            alpha = self.alpha
            beta = self.beta

        if len_effect and freq_effect:
            len_freq_input = stpp_input[:, :, -3:-1]
            len_freq_input_not_nan = torch.nan_to_num(len_freq_input, nan=0.0)
            alpha_effect_len_freq = self.alpha_len_freq(len_freq_input_not_nan)
            beta_effect_len_freq = self.beta_len_freq(len_freq_input_not_nan)
            location_effect_len_freq = self.loc_len_freq(len_freq_input_not_nan)

            if self.inter_reader_mark:
                int_len_reader_loc = (
                    self.len_reader_loc(reader_encoders)
                    * len_freq_input_not_nan[:, :, 0:1]
                )
                int_len_reader_alpha = (
                    self.len_reader_alpha(reader_encoders)
                    * len_freq_input_not_nan[:, :, 0:1]
                )
                int_len_reader_beta = (
                    self.len_reader_beta(reader_encoders)
                    * len_freq_input_not_nan[:, :, 0:1]
                )
                int_freq_reader_loc = (
                    self.freq_reader_loc(reader_encoders)
                    * len_freq_input_not_nan[:, :, 1:2]
                )
                int_freq_reader_alpha = (
                    self.freq_reader_alpha(reader_encoders)
                    * len_freq_input_not_nan[:, :, 1:2]
                )
                int_freq_reader_beta = (
                    self.freq_reader_beta(reader_encoders)
                    * len_freq_input_not_nan[:, :, 1:2]
                )
                location_output = (
                    location_output + int_len_reader_loc + int_freq_reader_loc
                )
                alpha = alpha + int_len_reader_alpha + int_freq_reader_alpha
                beta = beta + int_len_reader_beta + int_freq_reader_beta

            alpha = alpha + alpha_effect_len_freq
            beta = beta + beta_effect_len_freq
            location_output = location_output + location_effect_len_freq
            if word_effect and char_effect:
                word_effect = False
                char_effect = False
                past_fixation_mark = None
                cs_input = stpp_input[:, :, -5:-4]
                ws_input = stpp_input[:, :, -4:-3]

                cs_input_not_nan = torch.nan_to_num(cs_input, nan=0.0)
                ws_input_not_nan = torch.nan_to_num(ws_input, nan=0.0)
                beta_effect_ws = self.beta_ws(ws_input_not_nan)
                beta_effect_cs = self.beta_cs(cs_input_not_nan)
                alpha_effect_ws = self.alpha_ws(ws_input_not_nan)
                alpha_effect_cs = self.alpha_cs(cs_input_not_nan)
                location_effect_ws = self.loc_ws(ws_input_not_nan)
                location_effect_cs = self.loc_cs(cs_input_not_nan)

                if self.inter_reader_mark:
                    int_ws_reader_loc = (
                        self.ws_reader_loc(reader_encoders) * ws_input_not_nan
                    )
                    int_ws_reader_alpha = (
                        self.ws_reader_alpha(reader_encoders) * ws_input_not_nan
                    )
                    int_ws_reader_beta = (
                        self.ws_reader_beta(reader_encoders) * ws_input_not_nan
                    )

                    int_cs_reader_loc = (
                        self.cs_reader_loc(reader_encoders) * cs_input_not_nan
                    )
                    int_cs_reader_alpha = (
                        self.cs_reader_alpha(reader_encoders) * cs_input_not_nan
                    )
                    int_cs_reader_beta = (
                        self.cs_reader_beta(reader_encoders) * cs_input_not_nan
                    )
                    location_output = (
                        location_output + int_ws_reader_loc + int_cs_reader_loc
                    )
                    alpha = alpha + int_ws_reader_alpha + int_cs_reader_alpha
                    beta = beta + int_ws_reader_beta + int_cs_reader_beta

                alpha = alpha + alpha_effect_cs + alpha_effect_ws
                beta = beta + beta_effect_cs + beta_effect_ws
                location_output = (
                    location_output + location_effect_ws + location_effect_cs
                )

            if not (word_effect or char_effect):
                past_fixation_mark = None

                # Length and Frequency have predictors that are NA in the same exact spots. So we can just have one missing value term
                if self.missing_value_term_condition:
                    assert (
                        len_freq_input[:, :, 0:1].isnan()
                        == len_freq_input[:, :, 1:2].isnan()
                    ).all()

                    miss_location, miss_alpha, miss_beta = (
                        self.add_missing_value_term_mark(
                            len_freq_input[:, :, 0:1], reader_encoders
                        )
                    )
                    location_output = location_output + miss_location
                    alpha = alpha + miss_alpha

                    beta = beta + miss_beta

        if past_fixation_mark is not None:
            past_fixation_mark_and_not_nan = torch.nan_to_num(
                past_fixation_mark, nan=0.0
            )

            mark_effect_loc = self.location_mark(past_fixation_mark_and_not_nan)
            mark_effect_alpha = self.alpha_mark(past_fixation_mark_and_not_nan)
            mark_effect_beta = self.beta_mark(past_fixation_mark_and_not_nan)

            location_output = location_output + mark_effect_loc
            alpha = alpha + mark_effect_alpha
            beta = beta + mark_effect_beta

            if self.inter_reader_mark:
                mark_interaction_reader_loc = (
                    self.location_reader_inter(reader_encoders)
                    * past_fixation_mark_and_not_nan
                )
                mark_interaction_reader_alpha = (
                    self.alpha_reader_inter(reader_encoders)
                    * past_fixation_mark_and_not_nan
                )
                mark_interaction_reader_beta = (
                    self.beta_reader_inter(reader_encoders)
                    * past_fixation_mark_and_not_nan
                )
                location_output = location_output + mark_interaction_reader_loc
                alpha = alpha + mark_interaction_reader_alpha
                beta = beta + mark_interaction_reader_beta

        if (
            (char_effect or word_effect or len_effect or freq_effect)
            and self.missing_value_term_condition
            and past_fixation_mark is not None
        ):
            miss_location, miss_alpha, miss_beta = self.add_missing_value_term_mark(
                past_fixation_mark, reader_encoders
            )
            location_output = location_output + miss_location
            alpha = alpha + miss_alpha
            beta = beta + miss_beta

        alpha = F.relu(alpha)
        beta = F.relu(beta)
        sigma_value = F.relu(self.sigma) + 1e-9

        mu_value = F.relu(self.mu)

        return (mu_value, alpha, beta, sigma_value, location_output, None)

    def forward_duration(self, dur_input, curr_duration_time):
        onset_time_current_fixation = curr_duration_time[:, 1]

        dur_eff = "dur_conv" in self.duration_prediction_func
        char_eff = "char" in self.duration_prediction_func
        word_eff = "word" in self.duration_prediction_func
        len_eff = "len" in self.duration_prediction_func
        freq_eff = "freq" in self.duration_prediction_func
        duration_output = self.mu_duration

        if dur_eff == False:
            duration_spillover_input = dur_input[:, :, -1]
            dur_input = dur_input[:, :, :-1]
            mark_duration_onset_past_fix = dur_input[:, :, -1]
            gamma_alpha_dur = F.relu(self.duration_gamma_alpha) + 1 + 1e-3
            gamma_beta_dur = F.relu(self.duration_gamma_beta) + 1e-3
            gamma_delta_dur = F.relu(self.duration_gamma_delta) + 1e-3

            duration_spill_over_effects = gamma_convolution(
                input_feature=duration_spillover_input,
                input_feature_times=mark_duration_onset_past_fix,
                current_time=onset_time_current_fixation,
                alpha=gamma_alpha_dur,
                beta=gamma_beta_dur,
                delta=gamma_delta_dur,
                tracker=None,
                logger=self.logger,
            )
            duration_output = duration_output + self.duration_spillover_coeff(
                duration_spill_over_effects
            )

        mark_convol_input = None
        if dur_eff or char_eff or word_eff or len_eff or freq_eff:
            mark_convol_input = dur_input[:, :, READER_EMB_R_DUR]
            mark_duration_onset_past_fix = dur_input[:, :, READER_EMB_R_DUR + 1]

        if "reader" in self.duration_prediction_func:
            reader_encoder = dur_input[:, :, READER_EMB_L_DUR:READER_EMB_R_DUR]
            duration_output = duration_output + self.reader_linear_dur(
                reader_encoder[:, 0, :]
            )

        if len_eff and freq_eff:
            freq_input = dur_input[:, :, -2:-1].squeeze(-1)
            len_input = dur_input[:, :, -3:-2].squeeze(-1)
            mark_duration_onset_past_fix = dur_input[:, :, -1]
            freq_input_not_nan = torch.nan_to_num(freq_input, nan=0.0)
            len_input_not_nan = torch.nan_to_num(len_input, nan=0.0)
            len_freq_spillovers = self.get_spillover_effects_from_length_and_frequency(
                freq_input_not_nan=freq_input_not_nan,
                len_input_not_nan=len_input_not_nan,
                mark_duration_onset_past_fix=mark_duration_onset_past_fix,
                onset_time_current_fixation=onset_time_current_fixation,
            )

            next_fixation_index = torch.nonzero(
                onset_time_current_fixation.unsqueeze(-1)
                == mark_duration_onset_past_fix
            )
            next_frequency = freq_input[
                next_fixation_index[:, 0], next_fixation_index[:, 1]
            ].unsqueeze(-1)

            next_length = len_input[
                next_fixation_index[:, 0], next_fixation_index[:, 1]
            ].unsqueeze(-1)
            next_frequency_not_nan = torch.nan_to_num(next_frequency, nan=0.0)
            next_length_not_nan = torch.nan_to_num(next_length, nan=0.0)

            effect_next_freq = self.coeff_freq(next_frequency_not_nan)
            effect_next_len = self.coeff_len(next_length_not_nan)
            duration_output = (
                duration_output
                + effect_next_len
                + effect_next_freq
                + len_freq_spillovers
            )

            if self.inter_reader_mark:
                interaction_effect_next_freq = (
                    self.freq_interaction(reader_encoder[:, 0, :])
                    * next_frequency_not_nan
                )
                interaction_effect_next_len = (
                    self.len_interaction(reader_encoder[:, 0, :]) * next_length_not_nan
                )
                duration_output = (
                    duration_output
                    + interaction_effect_next_freq
                    + interaction_effect_next_len
                )

            # check if both word_eff and char_eff are true
            if word_eff and char_eff:
                word_eff = False
                char_eff = False
                # retrieve the input layer of word suprisal that should be in position -4
                # so the order in the input will be  char, word, len, freq, time
                word_input = dur_input[:, :, -4:-3].squeeze(-1)
                char_input = dur_input[:, :, -5:-4].squeeze(-1)
                word_input_not_nan = torch.nan_to_num(word_input, nan=0.0)
                char_input_not_nan = torch.nan_to_num(char_input, nan=0.0)
                word_char_spillovers = (
                    self.get_spillover_effects_from_char_and_word_surp(
                        char_input_not_nan=char_input_not_nan,
                        word_input_not_nan=word_input_not_nan,
                        mark_duration_onset_past_fix=mark_duration_onset_past_fix,
                        onset_time_current_fixation=onset_time_current_fixation,
                    )
                )
                next_fixation_index = torch.nonzero(
                    onset_time_current_fixation.unsqueeze(-1)
                    == mark_duration_onset_past_fix
                )
                next_word_surp = word_input[
                    next_fixation_index[:, 0], next_fixation_index[:, 1]
                ].unsqueeze(-1)

                next_char_surp = char_input[
                    next_fixation_index[:, 0], next_fixation_index[:, 1]
                ].unsqueeze(-1)
                next_word_surp_not_nan = torch.nan_to_num(next_word_surp, nan=0.0)
                next_char_surp_not_nan = torch.nan_to_num(next_char_surp, nan=0.0)

                effect_next_word = self.coeff_word_surp(next_word_surp_not_nan)
                effect_next_char = self.coeff_char_surp(next_char_surp_not_nan)
                duration_output = (
                    duration_output
                    + effect_next_word
                    + effect_next_char
                    + word_char_spillovers
                )

                if self.inter_reader_mark:
                    interaction_effect_next_word = (
                        self.ws_interaction(reader_encoder[:, 0, :])
                        * next_word_surp_not_nan
                    )
                    interaction_effect_next_char = (
                        self.cs_interaction(reader_encoder[:, 0, :])
                        * next_char_surp_not_nan
                    )
                    duration_output = (
                        duration_output
                        + interaction_effect_next_word
                        + interaction_effect_next_char
                    )
            if not (word_eff or char_eff):
                mark_convol_input = None
                if self.missing_value_term_condition:
                    missing_values = next_frequency.isnan().float()
                    missing_value_effect = self.linear_missing_value_coeff(
                        missing_values
                    )
                    if self.inter_reader_mark:
                        missing_value_effect += (
                            self.linear_missing_value_inter_coeff(
                                reader_encoder[:, 0, :]
                            )
                            * missing_values
                        )
                    duration_output = duration_output + missing_value_effect

        if mark_convol_input is not None:

            gamma_alpha = F.relu(self.gamma_alpha) + 1 + 1e-3
            gamma_beta = F.relu(self.gamma_beta) + 1e-3
            gamma_delta = F.relu(self.gamma_delta) + 1e-3
            mark_convol_input_and_not_nan = torch.nan_to_num(mark_convol_input, nan=0.0)

            conv_output = gamma_convolution(
                input_feature=mark_convol_input_and_not_nan,
                input_feature_times=mark_duration_onset_past_fix,
                current_time=onset_time_current_fixation,
                alpha=gamma_alpha,
                beta=gamma_beta,
                delta=gamma_delta,
                tracker=self.track_gamma_values,
                logger=self.logger,
            )

            duration_output = self.lin_coeff_convolution(conv_output) + duration_output

        if (mark_convol_input is not None) and (dur_eff == False):

            next_fixation_index = torch.nonzero(
                onset_time_current_fixation.unsqueeze(-1)
                == mark_duration_onset_past_fix
            )
            next_fixation_value = mark_convol_input[
                next_fixation_index[:, 0], next_fixation_index[:, 1]
            ].unsqueeze(-1)

            next_fixation_value_not_nan = torch.nan_to_num(next_fixation_value, nan=0.0)

            if next_fixation_value.shape[0] != mark_convol_input.shape[0]:
                raise ValueError(
                    "Mismatch in shape of next_fixation_value and mark_convol_input"
                )
            duration_output = duration_output + self.current_value_coeff(
                next_fixation_value_not_nan
            )

            if self.inter_reader_mark:
                interaction_effect_next_value = (
                    self.next_fixation_value_interaction(reader_encoder[:, 0, :])
                    * next_fixation_value_not_nan
                )
                duration_output = duration_output + interaction_effect_next_value

            if self.missing_value_term_condition:
                missing_values = next_fixation_value.isnan().float()
                missing_value_effect = self.linear_missing_value_coeff(missing_values)
                if self.inter_reader_mark:
                    missing_value_effect += (
                        self.linear_missing_value_inter_coeff(reader_encoder[:, 0, :])
                        * missing_values
                    )
                duration_output = duration_output + missing_value_effect

        # if 'dur_conv' not in self.duration_prediction_func:

        duration_output = F.relu(duration_output)
        variance = F.relu(self.sigma_duration)

        duration_output = (duration_output, variance)
        return (
            None,
            None,
            None,
            None,
            None,
            duration_output,
        )

    def initialize_parameters_duration(self):

        initial_alpha_value = float(self.cfg.alpha_g)
        initial_beta_value = float(self.cfg.beta_g)
        initial_delta_value = float(self.cfg.delta_g)

        # * --------------------------------*
        # * Duration Weights Initialization *
        # * --------------------------------*
        self.mu_duration = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.sigma_duration = nn.Parameter(torch.empty((), dtype=torch.float32))

        if self.cfg.division_factor_durations == 100:
            nn.init.constant_(self.mu_duration, 0.648438)
            nn.init.constant_(self.sigma_duration, 0.18)
        elif self.cfg.division_factor_durations == 1:
            nn.init.constant_(self.mu_duration, 5.253608)
            nn.init.constant_(self.sigma_duration, 0.18)

        duration_effects_dur = "dur" in self.duration_prediction_func
        wordsurp_effects_dur = "word" in self.duration_prediction_func
        charsurp_effects_dur = "char" in self.duration_prediction_func
        len_effects_dur = "len" in self.duration_prediction_func
        freq_effects_dur = "freq" in self.duration_prediction_func
        initialize_mark_weights = False
        if (
            duration_effects_dur
            or wordsurp_effects_dur
            or charsurp_effects_dur
            or len_effects_dur
            or freq_effects_dur
        ):
            initialize_mark_weights = True

        self.duration_gamma_alpha = nn.Parameter(
            torch.full((1,), initial_alpha_value),
        )
        self.duration_gamma_beta = nn.Parameter(
            torch.full((1,), initial_beta_value),
        )
        self.duration_gamma_delta = nn.Parameter(
            torch.full((1,), initial_delta_value),
        )

        self.duration_spillover_coeff = nn.Linear(1, 1, bias=False)

        if len_effects_dur and freq_effects_dur:
            self.freq_gamma_alpha = nn.Parameter(torch.full((1,), initial_alpha_value))
            self.freq_gamma_beta = nn.Parameter(torch.full((1,), initial_beta_value))
            self.freq_gamma_delta = nn.Parameter(torch.full((1,), initial_delta_value))

            self.ws_gamma_alpha = nn.Parameter(torch.full((1,), initial_alpha_value))
            self.ws_gamma_beta = nn.Parameter(torch.full((1,), initial_beta_value))
            self.ws_gamma_delta = nn.Parameter(torch.full((1,), initial_delta_value))

            self.cs_gamma_alpha = nn.Parameter(torch.full((1,), initial_alpha_value))
            self.cs_gamma_beta = nn.Parameter(torch.full((1,), initial_beta_value))
            self.cs_gamma_delta = nn.Parameter(torch.full((1,), initial_delta_value))

            self.len_gamma_alpha = nn.Parameter(torch.full((1,), initial_alpha_value))
            self.len_gamma_beta = nn.Parameter(torch.full((1,), initial_beta_value))
            self.len_gamma_delta = nn.Parameter(torch.full((1,), initial_delta_value))

            self.len_spillover_effects = nn.Linear(1, 1, bias=False)
            self.freq_spillover_effects = nn.Linear(1, 1, bias=False)
            self.word_surp_spillover_effects = nn.Linear(1, 1, bias=False)
            self.char_surp_spillover_effects = nn.Linear(1, 1, bias=False)

            self.coeff_word_surp = nn.Linear(1, 1, bias=False)
            self.coeff_char_surp = nn.Linear(1, 1, bias=False)
            self.coeff_freq = nn.Linear(1, 1, bias=False)
            self.coeff_len = nn.Linear(1, 1, bias=False)
            self.freq_interaction = nn.Linear(46, 1, bias=False)
            self.len_interaction = nn.Linear(46, 1, bias=False)
            self.ws_interaction = nn.Linear(46, 1, bias=False)
            self.cs_interaction = nn.Linear(46, 1, bias=False)

            if not (wordsurp_effects_dur or charsurp_effects_dur):
                initialize_mark_weights = False

        if initialize_mark_weights:
            self.lin_coeff_convolution = nn.Linear(1, 1, bias=True)
            nn.init.normal_(self.lin_coeff_convolution.weight, mean=0.0, std=0.001)
            nn.init.constant_(self.lin_coeff_convolution.bias, 0.0)
            self.gamma_alpha = nn.Parameter(
                torch.full((1,), initial_alpha_value),
            )
            self.gamma_beta = nn.Parameter(
                torch.full((1,), initial_beta_value),
            )
            self.gamma_delta = nn.Parameter(
                torch.full((1,), initial_delta_value),
            )

        cond = initialize_mark_weights or (len_effects_dur and freq_effects_dur)
        if self.inter_reader_mark and cond:

            if (
                "char" in self.duration_prediction_func
                or "word" in self.duration_prediction_func
                or "len" in self.duration_prediction_func
                or "freq" in self.duration_prediction_func
            ):
                self.next_fixation_value_interaction = nn.Linear(46, 1, bias=False)

        if "reader" in self.duration_prediction_func:
            self.reader_linear_dur = nn.Linear(46, 1, bias=True)
            nn.init.normal_(self.reader_linear_dur.weight, mean=0.0, std=0.001)
            nn.init.constant_(self.reader_linear_dur.bias, 0.0)

        if (
            wordsurp_effects_dur
            or charsurp_effects_dur
            or len_effects_dur
            or freq_effects_dur
        ):
            self.current_value_coeff = nn.Linear(1, 1, bias=False)

            if self.missing_value_term_condition:
                self.linear_missing_value_coeff = nn.Linear(1, 1, bias=False)
                self.linear_missing_value_inter_coeff = nn.Linear(46, 1, bias=False)

        self.gamma = GammaConvolution()

    def initialize_parameters_saccades(
        self,
    ):

        # * --------------------------
        # * HAWKS PARAMETERS
        # * --------------------------
        ####################################
        # Initialize POSITION LINEAR SHIFT
        ####################################
        self.position_shift = nn.Linear(2, 2, bias=True)
        torch.nn.init.normal_(self.position_shift.weight, mean=1, std=0.001)
        torch.nn.init.constant_(self.position_shift.bias, 0.0)

        ###########################################
        # Initialize Readers Embedding Layers     #
        ###########################################
        if "reader" in self.hawkes_predictors_func:

            self.reader_linear_hawks = nn.Linear(46, 2, bias=True)
            self.reader_linear_alpha = nn.Linear(46, 1, bias=True)
            self.reader_linear_beta = nn.Linear(46, 1, bias=True)

            #########################
            # Initialize Weights    #
            #########################
            nn.init.normal_(self.reader_linear_alpha.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.reader_linear_beta.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.reader_linear_hawks.weight, mean=0.0, std=0.001)

            #########################
            # Initialize Biases     #
            #########################

            # The biases of alpha and beta will be the initial global parameters of alpha and beta
            bias_alpha = torch.normal(
                mean=0,
                std=4,
                size=self.reader_linear_alpha.bias.shape,
                device=self.reader_linear_alpha.bias.device,
            )

            bias_beta = torch.normal(
                mean=0,
                std=7,
                size=self.reader_linear_beta.bias.shape,
                device=self.reader_linear_beta.bias.device,
            )
            self.reader_linear_alpha.bias.copy_(torch.abs(bias_alpha))
            self.reader_linear_beta.bias.copy_(torch.abs(bias_beta))
        else:

            self.alpha = nn.Parameter(torch.empty((), dtype=torch.float32))
            self.beta = nn.Parameter(torch.empty((), dtype=torch.float32))
            nn.init.constant_(self.beta, val=14)
            nn.init.constant_(self.alpha, val=8)
        self.sigma = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.mu = nn.Parameter(torch.empty((), dtype=torch.float32))
        nn.init.constant_(self.sigma, val=0.09)
        self.sigma.data.normal_(0, 0.001).abs_()
        self.mu.data.normal_(0, 0.05).abs_()

        # * -------------------------- *
        # * Hawkes Process Predictors  *
        # * -------------------------- *
        duration_effects_hp = "duration" in self.hawkes_predictors_func
        wordsurp_effects_hp = "word" in self.hawkes_predictors_func
        charsurp_effects_hp = "char" in self.hawkes_predictors_func
        len_effects_hp = "len" in self.hawkes_predictors_func
        freq_effects_hp = "freq" in self.hawkes_predictors_func

        if (
            duration_effects_hp
            or wordsurp_effects_hp
            or charsurp_effects_hp
            or len_effects_hp
            or freq_effects_hp
        ):
            self.alpha_mark = nn.Linear(1, 1, bias=True)
            self.beta_mark = nn.Linear(1, 1, bias=True)
            self.location_mark = nn.Linear(1, 2, bias=True)
            #  -------------------------- Initialize Weights -------------------------- #
            nn.init.normal_(self.alpha_mark.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.beta_mark.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.location_mark.weight, mean=0.0, std=0.001)
            nn.init.constant_(self.alpha_mark.bias, 0.0)
            nn.init.constant_(self.beta_mark.bias, 0.0)
            nn.init.constant_(self.location_mark.bias, 0.0)

            if self.inter_reader_mark:
                self.alpha_reader_inter = nn.Linear(46, 1, bias=False)
                self.beta_reader_inter = nn.Linear(46, 1, bias=False)
                self.location_reader_inter = nn.Linear(46, 2, bias=False)
                nn.init.normal_(self.alpha_reader_inter.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.beta_reader_inter.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.location_reader_inter.weight, mean=0.0, std=0.0001)

        if len_effects_hp and freq_effects_hp:
            self.alpha_len_freq = nn.Linear(2, 1, bias=False)
            nn.init.normal_(self.alpha_len_freq.weight, mean=0.0, std=0.001)
            self.beta_len_freq = nn.Linear(2, 1, bias=False)
            nn.init.normal_(self.beta_len_freq.weight, mean=0.0, std=0.001)
            self.loc_len_freq = nn.Linear(2, 2, bias=False)
            nn.init.normal_(self.loc_len_freq.weight, mean=0.0, std=0.001)
            if self.inter_reader_mark:
                self.len_reader_loc = nn.Linear(46, 2, bias=False)
                self.len_reader_alpha = nn.Linear(46, 1, bias=False)
                self.len_reader_beta = nn.Linear(46, 1, bias=False)
                self.freq_reader_loc = nn.Linear(46, 2, bias=False)
                self.freq_reader_alpha = nn.Linear(46, 1, bias=False)
                self.freq_reader_beta = nn.Linear(46, 1, bias=False)
                nn.init.normal_(self.freq_reader_loc.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.freq_reader_alpha.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.freq_reader_beta.weight, mean=0.0, std=0.0001)

                nn.init.normal_(self.len_reader_loc.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.len_reader_alpha.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.len_reader_beta.weight, mean=0.0, std=0.0001)

        if wordsurp_effects_hp and charsurp_effects_hp:
            self.beta_ws = nn.Linear(1, 1, bias=False)
            self.beta_cs = nn.Linear(1, 1, bias=False)
            self.alpha_ws = nn.Linear(1, 1, bias=False)
            self.alpha_cs = nn.Linear(1, 1, bias=False)
            self.loc_ws = nn.Linear(1, 2, bias=False)
            self.loc_cs = nn.Linear(1, 2, bias=False)
            nn.init.normal_(self.beta_ws.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.beta_cs.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.alpha_ws.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.alpha_cs.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.loc_ws.weight, mean=0.0, std=0.001)
            nn.init.normal_(self.loc_cs.weight, mean=0.0, std=0.001)
            if self.inter_reader_mark:
                self.ws_reader_loc = nn.Linear(46, 2, bias=False)
                self.ws_reader_alpha = nn.Linear(46, 1, bias=False)
                self.ws_reader_beta = nn.Linear(46, 1, bias=False)
                self.cs_reader_loc = nn.Linear(46, 2, bias=False)
                self.cs_reader_alpha = nn.Linear(46, 1, bias=False)
                self.cs_reader_beta = nn.Linear(46, 1, bias=False)

                nn.init.normal_(self.ws_reader_loc.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.ws_reader_alpha.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.ws_reader_beta.weight, mean=0.0, std=0.0001)

                nn.init.normal_(self.cs_reader_loc.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.cs_reader_alpha.weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.cs_reader_beta.weight, mean=0.0, std=0.0001)

        if (
            wordsurp_effects_hp
            or charsurp_effects_hp
            or len_effects_hp
            or freq_effects_hp
        ):
            if self.missing_value_term_condition:
                self.linear_missing_value_location = nn.Linear(1, 2, bias=False)
                self.linear_missing_value_alpha = nn.Linear(1, 1, bias=False)
                self.linear_missing_value_beta = nn.Linear(1, 1, bias=False)

                self.missing_value_location_reader = nn.Linear(46, 2, bias=False)
                self.missing_value_alpha_reader = nn.Linear(46, 1, bias=False)
                self.missing_value_beta_reader = nn.Linear(46, 1, bias=False)

                nn.init.normal_(
                    self.linear_missing_value_location.weight, mean=0.0, std=0.0001
                )
                nn.init.normal_(
                    self.linear_missing_value_alpha.weight, mean=0.0, std=0.0001
                )
                nn.init.normal_(
                    self.linear_missing_value_beta.weight, mean=0.0, std=0.0001
                )
                nn.init.normal_(
                    self.missing_value_location_reader.weight, mean=0.0, std=0.0001
                )
                nn.init.normal_(
                    self.missing_value_alpha_reader.weight, mean=0.0, std=0.0001
                )
                nn.init.normal_(
                    self.missing_value_beta_reader.weight, mean=0.0, std=0.0001
                )

    def add_missing_value_term_mark(self, value, reader_encoders):
        missing_value_vector = (torch.isnan(value)).float()
        missing_value_effect_location = (
            self.linear_missing_value_location(missing_value_vector)
            + self.missing_value_location_reader(reader_encoders) * missing_value_vector
        )
        missing_value_effect_alpha = (
            self.linear_missing_value_alpha(missing_value_vector)
            + self.missing_value_alpha_reader(reader_encoders) * missing_value_vector
        )
        missing_value_effect_beta = (
            self.linear_missing_value_beta(missing_value_vector)
            + self.missing_value_beta_reader(reader_encoders) * missing_value_vector
        )

        return (
            missing_value_effect_location,
            missing_value_effect_alpha,
            missing_value_effect_beta,
        )

    def get_spillover_effects_from_length_and_frequency(
        self,
        freq_input_not_nan,
        len_input_not_nan,
        mark_duration_onset_past_fix,
        onset_time_current_fixation,
    ):

        freq_gamma_alpha = F.relu(self.freq_gamma_alpha) + 1 + 1e-3
        freq_gamma_beta = F.relu(self.freq_gamma_beta) + 1e-3
        freq_gamma_delta = F.relu(self.freq_gamma_delta) + 1e-3

        len_gamma_alpha = F.relu(self.len_gamma_alpha) + 1 + 1e-3
        len_gamma_beta = F.relu(self.len_gamma_beta) + 1e-3
        len_gamma_delta = F.relu(self.len_gamma_delta) + 1e-3
        freq_effects = gamma_convolution(
            input_feature=freq_input_not_nan,
            input_feature_times=mark_duration_onset_past_fix,
            current_time=onset_time_current_fixation,
            alpha=freq_gamma_alpha,
            beta=freq_gamma_beta,
            delta=freq_gamma_delta,
            tracker=None,
            logger=self.logger,
            mark="freq",
        )
        len_effects = gamma_convolution(
            input_feature=len_input_not_nan,
            input_feature_times=mark_duration_onset_past_fix,
            current_time=onset_time_current_fixation,
            alpha=len_gamma_alpha,
            beta=len_gamma_beta,
            delta=len_gamma_delta,
            tracker=None,
            logger=self.logger,
            mark="len",
        )

        spillovers = self.len_spillover_effects(
            len_effects
        ) + self.freq_spillover_effects(freq_effects)

        return spillovers

    def get_spillover_effects_from_char_and_word_surp(
        self,
        char_input_not_nan,
        word_input_not_nan,
        mark_duration_onset_past_fix,
        onset_time_current_fixation,
    ):

        ws_gamma_alpha = F.relu(self.ws_gamma_alpha) + 1 + 1e-3
        ws_gamma_beta = F.relu(self.ws_gamma_beta) + 1e-3
        ws_gamma_delta = F.relu(self.ws_gamma_delta) + 1e-3

        cs_gamma_alpha = F.relu(self.cs_gamma_alpha) + 1 + 1e-3
        cs_gamma_beta = F.relu(self.cs_gamma_beta) + 1e-3
        cs_gamma_delta = F.relu(self.cs_gamma_delta) + 1e-3
        char_surp_effects = gamma_convolution(
            input_feature=char_input_not_nan,
            input_feature_times=mark_duration_onset_past_fix,
            current_time=onset_time_current_fixation,
            alpha=cs_gamma_alpha,
            beta=cs_gamma_beta,
            delta=cs_gamma_delta,
            tracker=None,
            logger=self.logger,
            mark="char",
        )
        word_surp_effects = gamma_convolution(
            input_feature=word_input_not_nan,
            input_feature_times=mark_duration_onset_past_fix,
            current_time=onset_time_current_fixation,
            alpha=ws_gamma_alpha,
            beta=ws_gamma_beta,
            delta=ws_gamma_delta,
            tracker=None,
            logger=self.logger,
            mark="word",
        )

        spillovers = self.word_surp_spillover_effects(
            word_surp_effects
        ) + self.char_surp_spillover_effects(char_surp_effects)

        return spillovers
