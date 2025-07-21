library(dplyr)
library(purrr)

# ── 1) Data prep ─────────────────────────────────────────────────────────────
library(rsample)
set.seed(20250604)
duration_spillover = FALSE

gaze_duration_dataset <- ".../fine-grained-model-reading-behaviour/data/meco_df_gaze_duration.csv"

gaze_dur_df <- read.csv(gaze_duration_dataset)


gaze_dur_df<- gaze_dur_df %>%
  rename_with(
    .cols = matches("^(prev2|prev)_"),
    .fn   = ~ sub("^prev2_(.*)$", "\\1_prev2",
                  sub("^prev_(.*)$",   "\\1_prev", .x))
  )

gaze_dur_df <- gaze_dur_df %>% rename( ws_prev = surp_prev,  ws_prev2 = surp_prev2, ws = word_level_surprisal)

response_col    <- "dur"
predictors_cols <- c("freq", "len", "ws")
index_cols      <- c("reader", "text", "fixid")

gaze_dur_df$reader <- as.factor(gaze_dur_df$reader)

gaze_dur_df <- gaze_dur_df  %>%  slice_sample(prop = 1, replace = FALSE) 

any(duplicated(gaze_dur_df[, c("reader", "text", "ianum_word")]))



logLik_newdata <- function(fit, newdata) {
  warned <- FALSE
  
  ll <- withCallingHandlers({
    
    mu  <- predict(fit, newdata)
    sd2 <-  sigma(fit)^2
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

eval_preds_gaze_dur <- function(pred_cols,
                                normalize                 = TRUE,   
                                adding_duration_spillovers = duration_spillover) {
  
  ## --------------------------------------------------------------------- ##
  ##  helpers                                                              ##
  ## --------------------------------------------------------------------- ##
  col_stats <- function(dat, cols) {
    # returns a named list: stats[["col"]] = c(min = <>, max = <>)
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
  
  ## --------------------------------------------------------------------- ##
  ##  cross-validation splits                                              ##
  ## --------------------------------------------------------------------- ##
  folds <- vfold_cv(gaze_dur_df, v = 5)
  
  ## --------------------------------------------------------------------- ##
  ##  formulas                                                             ##
  ## --------------------------------------------------------------------- ##
  all_terms <- purrr::map(pred_cols, \(p)
                          c(p, paste0(p, "_prev"), paste0(p, "_prev2"), "reader")
  ) |>
    unlist()
  interaction_terms <- paste0(pred_cols, ":reader")
  all_terms <- c(all_terms, interaction_terms)
  
  if (!adding_duration_spillovers) {
    fm_full <- as.formula(paste("dur ~", paste(all_terms, collapse = " + ")))
    fm_null <- dur ~ 1 + reader
    predictors_null <- "intercept + reader"
  } else {
    fm_full <- as.formula(
      paste("dur ~", paste(all_terms, collapse = " + "),
            "+ dur_prev + dur_prev2"))
    fm_null <- dur ~ 1 + reader + dur_prev + dur_prev2
    predictors_null <- "intercept + dur_prev + dur_prev2 + reader"
  }
  
  message("➡️  Full model formula: ", deparse(fm_full))
  message("➡️  Null model formula: ", deparse(fm_null))
  
  predictors_full <- paste(pred_cols, collapse = "+")
  norm_cols <- c(pred_cols,
                 paste0(pred_cols, "_prev"),
                 paste0(pred_cols, "_prev2"))
  
  ## --------------------------------------------------------------------- ##
  ##  5-fold CV                                                            ##
  ## --------------------------------------------------------------------- ##
  cv_res <- purrr::map_dfr(seq_len(5), function(k) {
    
    split <- folds$splits[[k]]
    train <- analysis(split)
    test  <- assessment(split)
    
    ## ----- (optional) normalisation ------------------------------------ ##
    if (normalize) {
      stats <- col_stats(train, norm_cols)
      train <- apply_normalise(train, stats)
      test  <- apply_normalise(test,  stats)
    }
    
    ## ----- fit & evaluate ---------------------------------------------- ##
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
      full_formula    = paste(deparse(fm_full), collapse = " "),
      null_formula    = paste(deparse(fm_null), collapse = " "),
      method          = "cross-validation 5-fold"
    ) |>
      dplyr::mutate(
        LLR             = 2 * (logLik_full - logLik_null),
        DeltaLogLikWord = (logLik_full - logLik_null) / n_test
      )
  })
  
  ## --------------------------------------------------------------------- ##
  ##  hold-out split                                                       ##
  ## --------------------------------------------------------------------- ##
  ho_train <- gaze_dur_df |>
    dplyr::filter(split == "train")
  ho_test  <- gaze_dur_df |>
    dplyr::filter(split == "test")
  
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
    predictors_full  = predictors_full,
    predictors_null  = predictors_null,
    full_formula     = paste(deparse(fm_full), collapse = " "),
    null_formula     = paste(deparse(fm_null), collapse = " "),
    method           = "split hold-out",
    n_test           = nrow(ho_test),
    logLik_full      = ho_full$logLik,
    logLik_null      = ho_null$logLik,
    warn_full        = ho_full$warned,
    warn_null        = ho_null$warned
  ) |>
    dplyr::mutate(
      LLR             = 2 * (logLik_full - logLik_null),
      DeltaLogLikWord = (logLik_full - logLik_null) / n_test
    )
  
  ## --------------------------------------------------------------------- ##
  ##  return                                                               ##
  ## --------------------------------------------------------------------- ##
  dplyr::bind_rows(cv_res, ho_res)
}

all_preds <- list(
  c("freq"),
  c("len"),
  c("ws"),
  c("freq", "len"),
  c("freq", "len", "ws")

)

results_gaze_dur <- map_dfr(all_preds, eval_preds_gaze_dur)


results_gaze_dur_summary <- results_gaze_dur %>%
  filter(!is.na(fold)) %>%             # just the 5 CV‐fold rows
  group_by(predictors_full,predictors_null, full_formula, null_formula) %>%        # one summary per experiment
  summarise(
    mean_Delta   = mean(DeltaLogLikWord, na.rm = TRUE),
    sd_Delta     = sd(DeltaLogLikWord,   na.rm = TRUE),
    min_Delta    = min(DeltaLogLikWord,  na.rm = TRUE),
    max_Delta    = max(DeltaLogLikWord,  na.rm = TRUE),
    n_test = mean(n_test),
    .groups = "drop"
  )

print(results_gaze_dur_summary[c(1,4,5,2,3), c(1,2,5,6,7,8,9,3,4)], width = 900)