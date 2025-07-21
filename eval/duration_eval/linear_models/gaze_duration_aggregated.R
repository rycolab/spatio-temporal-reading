library(dplyr)
library(purrr)
library(rsample)
library(tidyr)  


seed =20250604
set.seed(seed)


gaze_duration_dataset <- ".../fine-grained-model-reading-behaviour/data/meco_df_gaze_duration.csv"
skip_rate_ds <- ".../fine-grained-model-reading-behaviour/data/skip_rate_gaze_df.csv"

#To reproduce the experiments: 
# Aggregation GD (No Dur Spill)
aggregating_with_0s = FALSE
adding_duration_spillovers = FALSE
source_gaze_dur_ds = TRUE

# Aggregation GD (Dur Spill)
#aggregating_with_0s = FALSE
#adding_duration_spillovers = TRUE
#source_gaze_dur_ds = TRUE




if (aggregating_with_0s) {
  df <- read.csv(skip_rate_ds)
  
  df <- df %>%
    rename_with(
      .cols = matches("^(prev2|prev)_"),
      .fn = ~ sub("^prev2_(.*)$", "\\1_prev2",
                  sub("^prev_(.*)$", "\\1_prev", .x))
    ) %>%
    rename(ws_prev = surp_prev, ws_prev2 = surp_prev2, ws = surp)
  
} else if (!aggregating_with_0s && source_gaze_dur_ds) {
  df <- read.csv(gaze_duration_dataset)
  
  df <- df %>%
    rename_with(
      .cols = matches("^(prev2|prev)_"),
      .fn = ~ sub("^prev2_(.*)$", "\\1_prev2",
                  sub("^prev_(.*)$", "\\1_prev", .x))
    ) %>%
    rename(ws_prev = surp_prev, ws_prev2 = surp_prev2, ws = word_level_surprisal)
  
} else if (!aggregating_with_0s && !source_gaze_dur_ds) {
  message("Using Data Source from disaggregated with 0s")
  df <- read.csv(skip_rate_ds) %>%
    filter(dur != 0) %>%
    rename_with(
      .cols = matches("^(prev2|prev)_"),
      .fn = ~ sub("^prev2_(.*)$", "\\1_prev2",
                  sub("^prev_(.*)$", "\\1_prev", .x))
    ) %>%
    rename(ws_prev = surp_prev, ws_prev2 = surp_prev2, ws = surp)
}


index_cols <- c('text', 'ianum_word')
aggregate_col <- 'dur'

# 1. collect all the base and lag columns
word_based_cols <- c(
  "ws",
  "len",
  "freq",
  grep("(_prev2|_prev)$", names(df), value = TRUE)  
)

# 2. sanity‐check that they’re constant within each (text, ianum_word)
var_check <- df %>%
  group_by(text, ianum_word) %>%
  summarise(
    across(all_of(word_based_cols), ~ n_distinct(.x)),
    .groups = "drop"
  ) %>%
  filter(if_any(all_of(word_based_cols), ~ .x > 1))

if (nrow(var_check) > 0) {
  warning("Some groups have varying word‐based values:")
  print(var_check)
} else {
  message("All word‐based columns are fixed within each group.")
}

# 3. collapse to one row per (text, ianum_word)
df2 <- df %>%
  group_by(text, ianum_word) %>%
  summarise(
    across(all_of(word_based_cols), mean, na.rm = TRUE),
    dur = log(mean(exp(dur))),
    .groups = "drop"
  ) %>%  slice_sample(prop = 1, replace = FALSE) 


 

# Define the join keys
join_keys <- c("trialid", "ianum")




logLik_newdata <- function(fit, newdata) {
  warned <- FALSE
  
  ll <- withCallingHandlers({
    
    mu  <- predict(fit, newdata)
    sd2 <- sigma(fit)^2
    y   <- newdata[[as.character(formula(fit)[[2]])]]
    
    sum(dnorm(y, mu, sqrt(sd2), log = TRUE), na.rm = TRUE)
    
  }, warning = function(w) {
    if (grepl("rank-deficient", conditionMessage(w))) {
      warned <<- TRUE
    }
    invokeRestart("muffleWarning")
  })
  
  list(logLik = ll, warned = warned)
}

eval_preds_filtered <- function(pred_cols) {
  folds <- vfold_cv(df2, v = 5)
  

  
  if (adding_duration_spillovers == TRUE){
  # build the two formulas once
  all_terms       <- unlist(map(pred_cols, ~ c(.x, paste0(.x, "_prev"), paste0(.x, "_prev2"))))
  fm_full         <- as.formula(paste("dur ~", paste(all_terms, collapse = " + "), ' + dur_prev + dur_prev2'))
  

  fm_null          <- as.formula("dur ~ dur_prev + dur_prev2")
  predictors_null  <- "intercept + dur_spill"
  }else{
    all_terms       <- unlist(map(pred_cols, ~ c(.x, paste0(.x, "_prev"), paste0(.x, "_prev2"))))
    fm_full         <- as.formula(paste("dur ~", paste(all_terms, collapse = " + ")))
    
    
    fm_null          <- as.formula("dur ~ 1")
    predictors_null  <- "intercept"
}
  
  predictors_full <- paste(pred_cols, collapse = "+")

  base_preds <- pred_cols
  norm_cols  <- c(base_preds, paste0(base_preds, "_prev"), paste0(base_preds, "_prev2"))
  
  map_dfr(seq_len(5), function(k) {
    split <- folds$splits[[k]]
    train <- analysis(split)
    test  <- assessment(split)
    
    # print out what formulas we're about to fit
    message(sprintf("▶ Fold %d — fitting FULL model:   %s", k, deparse(fm_full)))
    message(sprintf("▶ Fold %d — fitting NULL model:   %s", k, deparse(fm_null)))
    

    
    n_before <- nrow(train)
    
    train_fit <- train %>%
      drop_na(all_of(paste0(pred_cols, "_prev")),
              all_of(paste0(pred_cols, "_prev2")))
    
    n_after <- nrow(train_fit)
    message(sprintf(
      "Fold %d: dropped %d rows (from %d → %d) due to missing lags",
      k, n_before - n_after, n_before, n_after
    ))
    # 1) Fit the full model
    fit_full <- lm(fm_full, data = train_fit)
    
    # 2) Figure out how many rows were dropped by lm()
    na_full        <- fit_full$na.action
    n_dropped_full <- if (is.null(na_full)) 0 else length(na_full)
    n_used_full    <- nobs(fit_full)
    n_train_fit    <- nrow(train_fit)
    
    message(sprintf(
      "Fold %d FULL model: used %d/%d rows (dropped %d)",
      k, n_used_full, n_train_fit, n_dropped_full
    ))
    
    # 3) (Optionally) do the same for the null model
    fit_null <- lm(fm_null, data = train_fit)
    na_null        <- fit_null$na.action
    n_dropped_null <- if (is.null(na_null)) 0 else length(na_null)
    n_used_null    <- nobs(fit_null)
    
    message(sprintf(
      "Fold %d NULL model: used %d/%d rows (dropped %d)",
      k, n_used_null, n_train_fit, n_dropped_null
    ))
    
    full_res <- logLik_newdata(fit_full, test)
    null_res <- logLik_newdata(fit_null, test)
    
    tibble(
      fold            = k,
      r_squared = summary(fit_full)$r.squared,
      n_test          = nrow(test),
      logLik_full     = full_res$logLik,
      logLik_null     = null_res$logLik,
      warn_full       = full_res$warned,
      warn_null       = null_res$warned,
      predictors_full = predictors_full,
      predictors_null = predictors_null,
      full_formula =  paste(deparse(fm_full), collapse = " "),
      null_formula = paste(deparse(fm_null), collapse = " "),   
      method          = "cross-validation 5-fold"
    ) %>%
      mutate(
        LLR             = 2 * (logLik_full - logLik_null),
        DeltaLogLikWord = (logLik_full - logLik_null) / n_test
      )
  })
}

# And now you can safely do:
all_preds <- list(
  "freq",
  "len",
  "ws",
  c("freq","len"),
  c("freq","len","ws")
)

results2 <- map_dfr(all_preds, eval_preds_filtered)

library(dplyr)

results2_summary <- results2 %>%
  filter(!is.na(fold)) %>%             # just the 5 CV‐fold rows
  group_by(predictors_full,predictors_null,full_formula, null_formula) %>%        # one summary per experiment
  summarise(
    mean_Delta   = mean(DeltaLogLikWord, na.rm = TRUE),
    sd_Delta     = sd(DeltaLogLikWord,   na.rm = TRUE),
    min_Delta    = min(DeltaLogLikWord,  na.rm = TRUE),
    max_Delta    = max(DeltaLogLikWord,  na.rm = TRUE),
    n_test = mean(n_test, na.rm= TRUE),
    r_squared = mean(r_squared),
    .groups = "drop"
  )

results2_summary
