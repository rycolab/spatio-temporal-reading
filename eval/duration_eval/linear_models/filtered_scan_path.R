library(dplyr)
library(purrr)
library(rsample)
library(tidyr)  
set.seed(20250604)

# ── 1) Data prep ─────────────────────────────────────────────────────────────

full_scanpath_meco_path <- ".../fine-grained-model-reading-behaviour/data/meco_df_raw.csv"
duration_spillovers = FALSE


index_cols      <- c("reader", "text", "fixid")
response_col    <- "dur"
predictors_cols <- c("freq", "len", "word_level_surprisal", "char_level_surp")
split_col       <- "split"

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

df2 <- read.csv(full_scanpath_meco_path) %>%
  mutate(reader = factor(reader)) %>%
  select(
    all_of(index_cols),
    all_of(response_col),
    all_of(predictors_cols),
    all_of(split_col)
  ) %>%
  # 1a) drop any row with missing word_level_surprisal
  filter(!is.na(word_level_surprisal)) %>%
  arrange(reader, text, fixid) %>%
  group_by(reader, text) %>%
  # 1b) compute 1- and 2-step lags for response + predictors
  mutate(
    across(
      all_of(c(response_col, predictors_cols)),
      list(
        prev  = ~lag(.x,  1),
        prev2 = ~lag(.x,  2)
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()

df2$reader <- as.factor(df2$reader)
df2 <- df2 %>%  slice_sample(prop = 1, replace = FALSE) 



eval_preds_filtered <- function(pred_cols,
                                normalize                  = FALSE,  
                                adding_duration_spillovers = duration_spillovers) {
  
  ## ── 1) cross-validation setup ─────────────────────────────────────────
  folds <- vfold_cv(df2, v = 5)        
  
  ## ── 2) helpers ────────────────────────────────────────────────────────
  lagged_terms <- function(p) c(p, paste0(p, "_prev"), paste0(p, "_prev2"))
  
  col_stats <- function(dat, cols) {
    # list of per-column c(min = .., max = ..)
    purrr::set_names(cols) |>
      purrr::map(\(c) {
        v <- dat[[c]]
        c(min = min(v, na.rm = TRUE),
          max = max(v, na.rm = TRUE))
      })
  }
  
  apply_normalise <- function(dat, stats) {
    for (c in names(stats)) {
      if (!c %in% names(dat)) next
      rng <- stats[[c]]["max"] - stats[[c]]["min"]
      if (rng == 0 || is.na(rng)) rng <- 1
      dat[[c]] <- (dat[[c]] - stats[[c]]["min"]) / rng
    }
    dat
  }
  
  ## ── 3) full & null formulas ───────────────────────────────────────────
  all_terms         <- unlist(purrr::map(pred_cols, lagged_terms))
  interaction_terms <- paste0(pred_cols, ":reader")
  
  if (adding_duration_spillovers) {
    fm_full <- as.formula(
      paste("dur ~",
            paste(c(all_terms, "reader", interaction_terms),
                  collapse = " + "),
            "+ dur_prev + dur_prev2")
    )
    fm_null         <- dur ~ 1 + reader + dur_prev + dur_prev2
    predictors_null <- "intercept + reader + spill"
  } else {
    fm_full <- as.formula(
      paste("dur ~",
            paste(c(all_terms, "reader", interaction_terms),
                  collapse = " + "))
    )
    fm_null         <- dur ~ 1 + reader
    predictors_null <- "intercept + reader"
  }
  
  predictors_full <- paste(c(pred_cols,
                             paste0(pred_cols, ":reader")), collapse = " + ")
  
  message("➡️  Full model formula: ", deparse(fm_full))
  message("➡️  Null model formula: ", deparse(fm_null))
  
  base_preds <- pred_cols
  norm_cols  <- c(base_preds,
                  paste0(base_preds, "_prev"),
                  paste0(base_preds, "_prev2"))
  
  ## ── 4) 5-fold cross-validation ────────────────────────────────────────
  cv_res <- purrr::map_dfr(seq_len(nrow(folds)), function(k) {
    
    split <- folds$splits[[k]]
    train <- analysis(split)
    test  <- assessment(split)
    
    if (normalize) {
      stats <- col_stats(train, norm_cols)
      train <- apply_normalise(train, stats)
      test  <- apply_normalise(test,  stats)
    }
    
    train_fit <- train |>
      tidyr::drop_na(dplyr::all_of(paste0(pred_cols, "_prev")),
                     dplyr::all_of(paste0(pred_cols, "_prev2")))
    
    fit_full <- lm(fm_full, data = train_fit)
    fit_null <- lm(fm_null, data = train_fit)
    
    full_res <- logLik_newdata(fit_full, test)
    null_res <- logLik_newdata(fit_null, test)
    
    tibble::tibble(
      fold            = k,
      n_test          = nrow(test),
      logLik_full     = full_res$logLik,
      logLik_null     = null_res$logLik,
      warn_full       = full_res$warned,
      warn_null       = null_res$warned,
      predictors_full = predictors_full,
      predictors_null = predictors_null,
      method          = "cross-validation 5-fold"
    ) |>
      dplyr::mutate(
        LLR             = 2 * (logLik_full - logLik_null),
        DeltaLogLikWord = (logLik_full - logLik_null) / n_test
      )
  })
  
  ## ── 5) hold-out split ────────────────────────────────────────────────
  ho_train <- df2 |> dplyr::filter(split == "train")
  ho_test  <- df2 |> dplyr::filter(split == "test")
  
  if (normalize) {
    stats <- col_stats(ho_train, norm_cols)
    ho_train <- apply_normalise(ho_train, stats)
    ho_test  <- apply_normalise(ho_test,  stats)
  }
  
  ho_train_fit <- ho_train |>
    tidyr::drop_na(dplyr::all_of(paste0(pred_cols, "_prev")),
                   dplyr::all_of(paste0(pred_cols, "_prev2")))
  
  ho_full <- logLik_newdata(lm(fm_full, data = ho_train_fit), ho_test)
  ho_null <- logLik_newdata(lm(fm_null, data = ho_train_fit), ho_test)
  
  ho_res <- tibble::tibble(
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

# ── 3) Run it ────────────────────────────────────────────────────────────────

all_preds <- list(
  c("freq"),
  c("len"),
  c("word_level_surprisal"),
  c("char_level_surp"),
  c("freq", "len"),
  c("freq", "len", "word_level_surprisal"),
  c("freq", "len", "char_level_surp"),
  c("freq", "len", "char_level_surp", "word_level_surprisal" )
  
  
)


results2 <- map_dfr(all_preds, eval_preds_filtered)

results2_summary <- results2 %>%
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

print(results2_summary)
