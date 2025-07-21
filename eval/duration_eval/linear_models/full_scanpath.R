library(dplyr)
library(purrr)
library(tidyr)
library(rsample)
set.seed(20250604)

full_scanpath_meco_path <- "/Users/francescoignaziore/Projects/fine-grained-model-reading-behaviour/data/meco_df_raw.csv"
duration_spillovers = TRUE

index_cols      <- c("reader", "text", "fixid")
response_col    <- "dur"
predictors_cols <- c("freq", "len", "word_level_surprisal", "char_level_surp")
split_col       <- "split"

df <- read.csv(full_scanpath_meco_path) %>%
  mutate(reader = factor(reader)) %>%
  select(
    all_of(index_cols),
    all_of(response_col),
    all_of(predictors_cols),
    all_of(split_col)
  ) %>%
  arrange(reader, text, fixid) %>%
  group_by(reader, text) %>%
  mutate(
    # 1) create NA-flags for each predictor
    across(
      all_of(predictors_cols),
      ~ is.na(.x),
      .names = "is_na_{.col}"
    ),
    # 2) replace any NA in predictors with zero
    across(
      all_of(predictors_cols),
      ~ coalesce(.x, 0)
    ),
    # 3) compute 1- and 2-step lags for response + all predictors
    across(
      all_of(c(response_col, predictors_cols)),
      list(
        prev  = ~ lag(.x, 1),
        prev2 = ~ lag(.x, 2)
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()

df$reader <- as.factor(df$reader)
df <- df %>%  slice_sample(prop = 1, replace = FALSE) 

#------ MODEL ------




# helper to compute logLik on new data
# 1) returns both the logLik *and* a warning flag
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
eval_preds <- function(pred_cols,
                       normalize                  = TRUE,   # turn scaling on/off
                       adding_duration_spillovers = duration_spillovers) {
  
  if (length(pred_cols) == 0)
    stop("pred_cols must contain at least one predictor")
  
  ## ── 0) Split set-up ────────────────────────────────────────────────────
  folds <- vfold_cv(df, v = 5)
  
  ## ── 1) helpers ────────────────────────────────────────────────────────
  # per-column min / max
  col_stats <- function(dat, cols, na_flag_col) {
    purrr::set_names(cols) |>
      purrr::map(function(cn) {
        v <- dat[[cn]]
        if (!is.null(na_flag_col) && na_flag_col %in% names(dat)) {
          v <- v[ !dat[[na_flag_col]] ]      # drop imputed zeros
          if (length(v) == 0) v <- dat[[cn]] # fall back if all were NA
        }
        c(min = min(v, na.rm = TRUE),
          max = max(v, na.rm = TRUE))
      })
  }
  
  # apply scaling
  apply_normalise <- function(dat, stats) {
    for (cn in names(stats)) {
      if (!cn %in% names(dat)) next
      rng <- stats[[cn]]["max"] - stats[[cn]]["min"]
      if (rng == 0 || is.na(rng)) rng <- 1
      dat[[cn]] <- (dat[[cn]] - stats[[cn]]["min"]) / rng
    }
    dat
  }
  
  ## ── 2) Formula construction ───────────────────────────────────────────
  lagged_terms <- function(p) c(p, paste0(p, "_prev"), paste0(p, "_prev2"))
  
  first_pred      <- pred_cols[1]
  na_flag_col     <- paste0("is_na_", first_pred)
  na_flag_inter   <- paste0(na_flag_col, ":reader")
  
  # base + lags + per-reader interactions for every predictor
  all_terms <- unlist(
    purrr::map(pred_cols, \(p) c(
      lagged_terms(p),
      paste0(p, ":reader")
    ))
  )
  
  # add the single NA flag (and its interaction) just once
  all_terms <- c(all_terms, "reader", na_flag_col, na_flag_inter)
  
  if (adding_duration_spillovers) {
    fm_full <- as.formula(
      paste("dur ~", paste(all_terms, collapse = " + "),
            "+ dur_prev + dur_prev2"))
    fm_null         <- dur ~ 1 + reader + dur_prev + dur_prev2
    predictors_null <- "intercept + reader + spill"
  } else {
    fm_full <- as.formula(
      paste("dur ~", paste(all_terms, collapse = " + ")))
    fm_null         <- dur ~ 1 + reader
    predictors_null <- "intercept + reader"
  }
  
  predictors_full <- paste(pred_cols, collapse = " + ")
  
  message("➡️  Full model formula: ", deparse(fm_full))
  message("➡️  Null model formula: ", deparse(fm_null))
  
  ## ── 3) Columns to scale ───────────────────────────────────────────────
  norm_cols <- c(pred_cols,
                 paste0(pred_cols, "_prev"),
                 paste0(pred_cols, "_prev2"))
  
  ## ── 4) 5-fold cross-validation ────────────────────────────────────────
  cv_res <- purrr::map_dfr(seq_len(5), function(k) {
    
    split <- folds$splits[[k]]
    train <- analysis(split)
    test  <- assessment(split)
    
    if (normalize) {
      stats <- col_stats(train, norm_cols, na_flag_col)
      train <- apply_normalise(train, stats)
      test  <- apply_normalise(test,  stats)
    }
    
    # drop rows without lagged values
    train_fit <- train |>
      tidyr::drop_na(dplyr::all_of(paste0(pred_cols, "_prev")),
                     dplyr::all_of(paste0(pred_cols, "_prev2")))
    
    fit_full <- lm(fm_full, data = train_fit)
    fit_null <- lm(fm_null, data = train_fit)
    
    full_res <- logLik_newdata(fit_full, test)
    null_res <- logLik_newdata(fit_null, test)
    
    tibble(
      fold            = k,
      n_test          = nrow(test),
      logLik_full     = full_res$logLik,
      logLik_null     = null_res$logLik,
      warn_full       = full_res$warned,
      warn_null       = null_res$warned,
      r_squared       = summary(fit_full)$r.squared,
      predictors_full = predictors_full,
      predictors_null = predictors_null,
      method          = "cross-validation 5-fold"
    ) |>
      dplyr::mutate(
        LLR             = 2 * (logLik_full - logLik_null),
        DeltaLogLikWord = (logLik_full - logLik_null) / n_test
      )
  })
  
  ## ── 5) Hold-out split evaluation ──────────────────────────────────────
  ho_train <- df |> dplyr::filter(split == "train")
  ho_test  <- df |> dplyr::filter(split == "test")
  
  if (normalize) {
    stats <- col_stats(ho_train, norm_cols, na_flag_col)
    ho_train <- apply_normalise(ho_train, stats)
    ho_test  <- apply_normalise(ho_test,  stats)
  }
  
  ho_full <- logLik_newdata(lm(fm_full, data = ho_train), ho_test)
  ho_null <- logLik_newdata(lm(fm_null, data = ho_train), ho_test)
  
  ho_res <- tibble(
    predictors_full = predictors_full,
    predictors_null = predictors_null,
    method          = "split hold-out",
    n_test          = nrow(ho_test),
    logLik_full     = ho_full$logLik,
    logLik_null     = ho_null$logLik,
    warn_full       = ho_full$warned,
    warn_null       = ho_null$warned
  ) |>
    dplyr::mutate(
      LLR             = 2 * (logLik_full - logLik_null),
      DeltaLogLikWord = (logLik_full - logLik_null) / n_test
    )
  
  ## ── 6) return results ────────────────────────────────────────────────
  dplyr::bind_rows(cv_res, ho_res)
}


# apply to your four predictors + “dur” itself (as “previous-duration”)
all_preds <- list(
  c("freq"),
  c("len"),
  c('word_level_surprisal'),
  c("char_level_surp"),
  c("freq", "len") ,
c("freq", "len", "word_level_surprisal"),
c("freq", "len", "char_level_surp"),
c("freq", "len", "char_level_surp", 'word_level_surprisal'))
  



results2 <- map_dfr(
  all_preds,
  eval_preds
)
results_summary <- results2 %>%
  filter(!is.na(fold)) %>%             # just the 5 CV‐fold rows
  group_by(predictors_full,predictors_null) %>%        # one summary per experiment
  summarise(
    mean_Delta   = mean(DeltaLogLikWord, na.rm = TRUE),
    sd_Delta     = sd(DeltaLogLikWord,   na.rm = TRUE),
    min_Delta    = min(DeltaLogLikWord,  na.rm = TRUE),
    max_Delta    = max(DeltaLogLikWord,  na.rm = TRUE),
    n_test = mean(n_test),
    .groups = "drop"
  )

