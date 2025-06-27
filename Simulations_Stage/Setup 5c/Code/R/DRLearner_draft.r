# DR-Learner in R: two functions, one for hyperparameter tuning, one for estimation

# Required libraries
library(caret)        # for model tuning & data splitting

if (!require("randomForest")) {
  install.packages("randomForest")
}

# Install tidyverse if not already installed
if (!require("tidyverse")) {
  install.packages("tidyverse")
}

# Install rlearner from GitHub if not already installed
if (!require("rlearner")) {
  devtools::install_github("xnie/rlearner", upgrade = "never")
}


library(tidyverse)

library(rlearner)
library(randomForest) # for Random Forest implementation (via caret)

source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5/Code/R/function_hypparam_optimisation.R")

BScore <- function(x, y) mean((100*x - 100*y)^2)
bias <- function(x, y) mean(100*x - 100*y)
PEHE <- function(x, y) sqrt(mean((100*x - 100*y)^2))

MSE_100 <- function(x, y) sqrt(mean((100*x - 100*y)^2))

MC_se <- function(x, B) qt(0.975, B - 1) * sd(x) / sqrt(B)

r_loss <- function(y, mu, z, pi, tau) mean(((y - mu) - (z - pi) * tau)^2)

# Préparation des données -------------------------------------------------

# Importer les données
# Load Data
#curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
#setwd(curr_dir); setwd('./../../')

setwd("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5c")

#list_size <- c(500,1000,5000, 10000)


# Importer les hyperparamètres
hyperparams <- read.csv("./../Setup 1a/Data/hyperparams.csv")

data_train_test <- read.csv("./../Setup 1a/Data/simulated_1M_data.csv")
set.seed(502 + 5); seed = 502 + 5

cat("\n")
data_validation <- read.csv("./../Setup 1a/Data/simulated_10K_data_validation.csv")
size_sample_val = nrow(data_validation)
print("size_sample_val")
print(size_sample_val)
res_val = prepare_train_data(data = data_validation, hyperparams = hyperparams,
                             size_sample = size_sample_val,
                             train_ratio = 0)

val_augmX = res_val$test_augmX; z_val = res_val$z_test; y_val = res_val$y_test
val_CATT = res_val$Test_CATT


size_sample_train_test = 1000000
print("size_sample_train_test")
print(size_sample_train_test)
res_train_test = prepare_train_data(data = data_train_test, hyperparams = hyperparams,
                                    size_sample = size_sample_train_test,
                                    train_ratio = 0.7, seed = seed, verbose = TRUE)

train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
Test_CATT = res_train_test$Test_CATT


# dr_RF_learner_fit: fit submodels given one flat params list
dr_RF_learner_fit <- function(X, y, t, params = list(
                                  mtry            = 1,
                                  sample.fraction = 0.5,
                                  nodesizeSpl     = 3,
                                  nodesizeAvg     = 3,
                                  ntree           = 500
                                ), seed = 42) {
  set.seed(seed)
  # honest 50/50 split
  idx   <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr  <- X[idx, ];    y_tr  <- y[idx];    t_tr  <- t[idx]
  X_md  <- X[-idx, ];   y_md  <- y[-idx];   t_md  <- t[-idx]

  # unpack params
  ntree <- params$ntree
  mtry  <- params$mtry
  samp  <- floor(params$sample.fraction * nrow(X_tr))
  nspl  <- params$nodesizeSpl
  navg  <- params$nodesizeAvg

  # 1) propensity
  pi_mod <- randomForest(as.factor(t_tr) ~ ., data = X_tr,
                         ntree = ntree,
                         mtry  = mtry,
                         sampsize = samp,
                         nodesize = nspl)
  pihat  <- pmin(pmax(predict(pi_mod, X_md, type = "prob")[,2],1e-3),1-1e-3)

  # 2) mu0, mu1
  mu0_mod <- randomForest(y_tr[t_tr==0] ~ ., data = X_tr[t_tr==0, ],
                          ntree = ntree,
                          mtry  = mtry,
                          sampsize = floor(params$sample.fraction * sum(t_tr==0)),
                          nodesize = nspl)
  mu1_mod <- randomForest(y_tr[t_tr==1] ~ ., data = X_tr[t_tr==1, ],
                          ntree = ntree,
                          mtry  = mtry,
                          sampsize = floor(params$sample.fraction * sum(t_tr==1)),
                          nodesize = nspl)

  mu0hat <- predict(mu0_mod, X_md)
  mu1hat <- predict(mu1_mod, X_md)

  # 3) DR pseudo-outcome
  Z <- t_md; Y <- y_md
  dr_ps <- (Z/pihat - (1-Z)/(1-pihat)) * Y +
           ((1 - Z/pihat)*mu1hat - (1 - (1-Z)/(1-pihat))*mu0hat)

  # 4) final CATE model
  cate_mod <- randomForest(dr_ps ~ ., data = X_md,
                           ntree = ntree,
                           mtry  = mtry,
                           sampsize = floor(params$sample.fraction * nrow(X_md)),
                           nodesize = navg)

  list(pi_model = pi_mod,
       mu0_model = mu0_mod,
       mu1_model = mu1_mod,
       cate_model = cate_mod)
}

# dr_RF_learner_predict: extract cate predictions
dr_RF_learner_predict <- function(fit, newdata) {
  predict(fit$cate_model, newdata)
}

# dr_RF_learner_optimize: one‐by‐one tuning like X_RF
dr_RF_learner_optimize <- function(train_X, train_y, train_t,
                                val_X, val_CATT,
                                param_ranges = list(
                                  mtry            = c(1,2,3,4),
                                  sample.fraction = c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6),
                                  nodesizeSpl     = c(1,3,5,10,15,20,25,30),
                                  nodesizeAvg     = c(1,3,5,10,15,20,25,30),
                                  ntree           = c(300,500,1000,1500,2000,3000,5000,10000)
                                ),
                                seed = 42,
                                verbose = TRUE) {
  set.seed(seed)
  # default starting point
  best_params <- list(
    ntree = 1000,
    mtry  = round(ncol(train_X)*0.5),
    nodesizeSpl = 2,
    nodesizeAvg = 1,
    sample.fraction = 0.5
  )
  best_pehe <- Inf

  for (param in names(param_ranges)) {
    if (verbose) message("Tuning ", param)
    for (val in param_ranges[[param]]) {
      trial <- best_params
      trial[[param]] <- val

      fit <- tryCatch(
        dr_RF_learner_fit(train_X, train_y, train_t, params = trial, seed = seed),
        error = function(e) { if (verbose) message("  fit failed: ", e$message); NULL }
      )
      if (is.null(fit)) next

      preds <- tryCatch(
        dr_RF_learner_predict(fit, newdata = val_X),
        error = function(e) { if (verbose) message("  pred failed: ", e$message); NULL }
      )
      if (is.null(preds)) next

      pehe <- PEHE(val_CATT, preds)
      if (verbose) message("  ", param, "=", val, " → PEHE=", round(pehe,4))

      if (!is.na(pehe) && pehe < best_pehe) {
        best_pehe   <- pehe
        best_params <- trial
        if (verbose) message("  ✅ new best PEHE=", round(best_pehe,4))
      }
    }
  }

  best_params
}
#' Predict CATE using fitted DR-Learner
#'
#' @param fit_result output of dr_RF_learner_fit
#' @param newdata data.frame for prediction
#' @return numeric CATE predictions
#'
dr_RF_learner_predict <- function(fit_result, newdata) {
  predict(fit_result$cate_model, newdata)
}

# ─── 1) RA‐Learner ──────────────────────────────────────────────────────────

# Fit RA‐Learner
RA_RF_learner_fit <- function(X, y, w, params, seed = 42) {
  set.seed(seed)
  # honest split
  idx  <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr <- X[idx, ];     y_tr <- y[idx];     w_tr <- w[idx]
  X_md <- X[-idx, ];    y_md <- y[-idx];    w_md <- w[-idx]

  # unpack
  ntree <- params$ntree
  mtry  <- params$mtry
  samp_tr <- floor(params$sample.fraction * nrow(X_tr))
  samp_md <- floor(params$sample.fraction * nrow(X_md))
  nspl  <- params$nodesizeSpl
  navg  <- params$nodesizeAvg

  # 1) fit outcome models
  mu0 <- randomForest(y_tr[w_tr==0] ~ ., data = X_tr[w_tr==0, ],
                      ntree = ntree, mtry = mtry,
                      sampsize = floor(params$sample.fraction*sum(w_tr==0)),
                      nodesize = nspl)
  mu1 <- randomForest(y_tr[w_tr==1] ~ ., data = X_tr[w_tr==1, ],
                      ntree = ntree, mtry = mtry,
                      sampsize = floor(params$sample.fraction*sum(w_tr==1)),
                      nodesize = nspl)

  # 2) build RA pseudo-outcome on meta-sample
  mu0hat <- predict(mu0, X_md)
  mu1hat <- predict(mu1, X_md)
  W <- w_md;  Y <- y_md
  RA_RF_pseudo <- W * (Y - mu0hat) + (1 - W) * (mu1hat - Y)

  # 3) final regressor
  cate_mod <- randomForest(RA_RF_pseudo ~ ., data = X_md,
                           ntree = ntree, mtry = mtry,
                           sampsize = samp_md, nodesize = navg)

  list(mu0_model = mu0, mu1_model = mu1, cate_model = cate_mod)
}

# Predict with RA‐Learner
RA_RF_learner_predict <- function(fit, newdata) {
  predict(fit$cate_model, newdata)
}

# Optimize RA‐Learner by PEHE on a validation set
RA_RF_learner_optimize <- function(train_X, train_y, train_w,
                                val_X, val_CATT,
                                param_ranges = list(
                                  mtry            = c(1,2,3,4),
                                  sample.fraction = c(0.05,0.1,0.2),
                                  nodesizeSpl     = c(1,5,10),
                                  nodesizeAvg     = c(1,5,10),
                                  ntree           = c(500,1000,2000)
                                ),
                                seed = 42, verbose = TRUE) {
  set.seed(seed)
  # default
  best_params <- list(
    ntree = 1000,
    mtry  = round(ncol(train_X)/2),
    sample.fraction = 0.5,
    nodesizeSpl     = 5,
    nodesizeAvg     = 5
  )
  best_pehe <- Inf

  for (param in names(param_ranges)) {
    if (verbose) message("Tuning RA: ", param)
    for (value in param_ranges[[param]]) {
      trial <- best_params
      trial[[param]] <- value

      fit <- tryCatch(
        RA_RF_learner_fit(train_X, train_y, train_w, params = trial, seed = seed),
        error = function(e) { if(verbose) message("  fit failed: ",e$message); NULL }
      )
      if (is.null(fit)) next

      preds <- tryCatch(
        RA_RF_learner_predict(fit, newdata = val_X),
        error = function(e) { if(verbose) message("  pred failed: ",e$message); NULL }
      )
      if (is.null(preds)) next

      pehe <- PEHE(val_CATT, preds)
      if (verbose) message("  ",param,"=",value," → PEHE=",round(pehe,4))

      if (!is.na(pehe) && pehe < best_pehe) {
        best_pehe    <- pehe
        best_params  <- trial
        if (verbose) message("   ✅ new best PEHE=",round(pehe,4))
      }
    }
  }

  best_params
}

# ─── 2) PW‐Learner ──────────────────────────────────────────────────────────

# Fit PW‐Learner
PW_RF_learner_fit <- function(X, y, w, params, seed = 42) {
  set.seed(seed)
  idx  <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr <- X[idx, ];     y_tr <- y[idx];     w_tr <- w[idx]
  X_md <- X[-idx, ];    y_md <- y[-idx];    w_md <- w[-idx]

  # unpack
  ntree <- params$ntree
  mtry  <- params$mtry
  samp_tr <- floor(params$sample.fraction * nrow(X_tr))
  samp_md <- floor(params$sample.fraction * nrow(X_md))
  nspl  <- params$nodesizeSpl
  navg  <- params$nodesizeAvg

  # 1) propensity
  pi_mod <- randomForest(as.factor(w_tr) ~ ., data = X_tr,
                         ntree = ntree, mtry = mtry,
                         sampsize = samp_tr, nodesize = nspl)
  pihat  <- pmin(pmax(predict(pi_mod, X_md, type="prob")[,2],1e-3),1-1e-3)

  # 2) PW pseudo-outcome
  W <- w_md; Y <- y_md
  PW_RF_pseudo <- (W/pihat - (1 - W)/(1 - pihat)) * Y

  # 3) final regressor
  cate_mod <- randomForest(PW_RF_pseudo ~ ., data = X_md,
                           ntree = ntree, mtry = mtry,
                           sampsize = samp_md, nodesize = navg)

  list(pi_model = pi_mod, cate_model = cate_mod)
}

# Predict with PW‐Learner
PW_RF_learner_predict <- function(fit, newdata) {
  predict(fit$cate_model, newdata)
}

# Optimize PW‐Learner by PEHE
PW_RF_learner_optimize <- function(train_X, train_y, train_w,
                                val_X, val_CATT,
                                param_ranges = list(
                                  mtry            = c(1,2,3,4),
                                  sample.fraction = c(0.05,0.1,0.2),
                                  nodesizeSpl     = c(1,5,10),
                                  nodesizeAvg     = c(1,5,10),
                                  ntree           = c(500,1000,2000)
                                ),
                                seed = 42, verbose = TRUE) {
  set.seed(seed)
  best_params <- list(
    ntree = 1000,
    mtry  = round(ncol(train_X)/2),
    sample.fraction = 0.5,
    nodesizeSpl     = 5,
    nodesizeAvg     = 5
  )
  best_pehe <- Inf

  for (param in names(param_ranges)) {
    if (verbose) message("Tuning PW: ", param)
    for (value in param_ranges[[param]]) {
      trial <- best_params
      trial[[param]] <- value

      fit <- tryCatch(
        PW_RF_learner_fit(train_X, train_y, train_w, params = trial, seed = seed),
        error = function(e) { if(verbose) message("  fit failed: ",e$message); NULL }
      )
      if (is.null(fit)) next

      preds <- tryCatch(
        PW_RF_learner_predict(fit, newdata = val_X),
        error = function(e) { if(verbose) message("  pred failed: ",e$message); NULL }
      )
      if (is.null(preds)) next

      pehe <- PEHE(val_CATT, preds)
      if (verbose) message("  ",param,"=",value," → PEHE=",round(pehe,4))

      if (!is.na(pehe) && pehe < best_pehe) {
        best_pehe   <- pehe
        best_params <- trial
        if (verbose) message("   ✅ new best PEHE=",round(pehe,4))
      }
    }
  }

  best_params
}


if (1){
start_time <- Sys.time()
model <- rlasso(x = train_augmX[, -ncol(train_augmX)], w = z_train, y = y_train,
                     lambda_choice = "lambda.min", rs = FALSE)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("rlasso_execution_time : ")
print(execution_time)
test_est = predict(model, test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE_rlasso = PEHE(Test_CATT, test_est[z_test == 1])
print("rlasso - Perf on test data")
print(CATT_Test_PEHE_rlasso)
}
# ─── USAGE ─────────────────────────────────────────────────────
# param_ranges <- list(
#  mtry = c(1,2,3,4), sample.fraction = c(0.01,0.05,...),
#  nodesizeSpl = ..., nodesizeAvg = ..., ntree = ...
# )
print(1)
if (1){
    start_time <- Sys.time()
best_params <- dr_RF_learner_optimize(
  train_X   = train_augmX[,-ncol(train_augmX)],
  train_y   = y_train,
  train_t   = z_train,
  val_X     = val_augmX[z_val == 1, ],
  val_CATT  = val_CATT,
  param_ranges <- list(
    mtry = c(1, 2, 3, 4),
    sample.fraction = c(0.00001,0.00005,0.0001, 0.0005, 0.001,0.005, 0.01,0.1),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  ),
  verbose = TRUE
)
end_time <- Sys.time()
execution_time <- end_time - start_time
print("DR-RF opti_time : ")
print(execution_time)
}
print(2)
# fit_res     <- dr_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, t = z_train)
fit_res     <- dr_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, t = z_train, params = best_params)
print(3)
cate_preds  <- dr_RF_learner_predict(fit_res, newdata = test_augmX[,-ncol(test_augmX)])
print(4)
CATT_Test_PEHE_DR = PEHE(Test_CATT, cate_preds[z_test == 1])
print("DR - Perf on test data")
print(CATT_Test_PEHE_DR)



# RA:
best_ra <- RA_RF_learner_optimize(
  train_X   = train_augmX[,-ncol(train_augmX)],
  train_y   = y_train,
  train_w   = z_train,
  val_X     = val_augmX[z_val == 1, ],
  val_CATT  = val_CATT,
  param_ranges <- list(
    mtry = c(1, 2, 3, 4),
    sample.fraction = c(0.00001,0.00005,0.0001, 0.0005, 0.001,0.005, 0.01,0.1, 0.2, 0.3, 0.5),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  ),
  verbose = TRUE
)
fit_ra  <- RA_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, w = z_train, params = best_params)
pred_ra <- RA_RF_learner_predict(fit_ra, newdata = test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE_RA = PEHE(Test_CATT, pred_ra[z_test == 1])
print("RA - Perf on test data")
print(CATT_Test_PEHE_RA)

# PW:
if (0){
best_pw <- PW_RF_learner_optimize(
  train_X   = train_augmX[,-ncol(train_augmX)],
  train_y   = y_train,
  train_w   = z_train,
  val_X     = val_augmX[z_val == 1, ],
  val_CATT  = val_CATT,
  param_ranges <- list(
    mtry = c(1, 2, 3, 4),
    sample.fraction = c(0.00001,0.00005,0.0001, 0.0005, 0.001,0.005, 0.01,0.1),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  ),
  verbose = TRUE
)
fit_pw  <- PW_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, w = z_train, params = best_params)
pred_pw <- PW_RF_learner_predict(fit_pw, newdata = test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE_PW = PEHE(Test_CATT, pred_pw[z_test == 1])
print("PW - Perf on test data")
print(CATT_Test_PEHE_PW)
}

cat("\n\n\n\n\n")
print("rlasso - Perf on test data")
print(CATT_Test_PEHE_rlasso)
print("DR - Perf on test data")
print(CATT_Test_PEHE_DR)
print("RA - Perf on test data")
print(CATT_Test_PEHE_RA)
#print("PW - Perf on test data")
#print(CATT_Test_PEHE_PW)