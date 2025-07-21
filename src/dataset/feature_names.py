###################################
#     FEATURE NAMES & INDICES     #
###################################


####################################
# PREDICTOR INDICES SACCADE MODEL  #
####################################


LOCATION_L = 0
LOCATION_R = 2
MARGIN_L = LOCATION_R  # 2
MARGIN_R = MARGIN_L + 0  # 2 we are not using this eventually
READER_EMB_L = MARGIN_R  # 2
READER_EMB_R = READER_EMB_L + 46  # 48

DURATION_L = READER_EMB_R  # 48
DURATION_R = DURATION_L + 1  # 49
CHAR_LEVEL_SURP_L = READER_EMB_R  # 48
CHAR_LEVEL_SURP_R = CHAR_LEVEL_SURP_L + 1  # 49

WORD_LEVEL_SURP_L = READER_EMB_R  # 48
WORD_LEVEL_SURP_R = WORD_LEVEL_SURP_L + 1  # 49

assert (
    WORD_LEVEL_SURP_L == CHAR_LEVEL_SURP_L == DURATION_L
), "Word and char level surp should be at the same index"

MARK_INDEX_L = WORD_LEVEL_SURP_L

assert (
    WORD_LEVEL_SURP_R == CHAR_LEVEL_SURP_R == DURATION_R
), "Word and char level surp should be at the same index"

MARK_INDEX_R = WORD_LEVEL_SURP_R


#################################
# PREDICTOR INDICES DUR MODEL   #
#################################

READER_EMB_L_DUR = 0
READER_EMB_R_DUR = 46


FORWARD_FIXATION_SUBSET_NAME = "forward_fixation"
SAME_WORD_FIXATION_SUBSET_NAME = "same_word_fixation"
BACKWARD_FIXATION_SUBSET_NAME = "backward_fixation"
NEW_LINE_FIXATION_SUBSET_NAME = "new_line_fixation"
FIRST_WORD_FIXATION_SUBSET_NAME = "first_word_fixation"
BACKWARD_PRECISE_FIXATION_SUBSET_NAME = "backward_fixation_same_line_on_words"
FORWARD_PRECISE_FIXATION_SUBSET_NAME = "forward_fixation_same_line_on_words"

SUBSETS_MASKS = {
    FORWARD_FIXATION_SUBSET_NAME: 0,
    SAME_WORD_FIXATION_SUBSET_NAME: 1,
    BACKWARD_FIXATION_SUBSET_NAME: 2,
    NEW_LINE_FIXATION_SUBSET_NAME: 3,
    FIRST_WORD_FIXATION_SUBSET_NAME: 4,
    BACKWARD_PRECISE_FIXATION_SUBSET_NAME: 5,
    FORWARD_PRECISE_FIXATION_SUBSET_NAME: 6,
}
