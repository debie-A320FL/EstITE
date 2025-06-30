# DR-Learner in R: two functions, one for hyperparameter tuning, one for estimation

# Required libraries

if (!require("nnet")) {
  install.packages("nnet")
}

library(nnet)        # for model tuning & data splitting

if (!require("caret")) {
  install.packages("caret")
}

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

source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5c/Code/R/function_hypparam_optimisation.R")

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

binary = TRUE

param_ranges = list(
    sample.fraction = c(0.00001,0.00005,0.0001, 0.0005, 0.001,0.005, 0.01,0.1, 0.2, 0.3, 0.5),
    mtry = c(1, 2, 3, 4),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  )

cat("\n")
data_validation <- read.csv("./../Setup 1a/Data/simulated_10K_data_validation.csv")
size_sample_val = nrow(data_validation)
print("size_sample_val")
print(size_sample_val)
res_val = prepare_train_data(data = data_validation, hyperparams = hyperparams,
                             size_sample = size_sample_val,
                             train_ratio = 0, treatment_percentile = 10,binary=binary)

val_augmX = res_val$test_augmX; z_val = res_val$z_test; y_val = res_val$y_test
val_CATT = res_val$Test_CATT


size_sample_train_test = 1e5
print("size_sample_train_test")
print(size_sample_train_test)
res_train_test = prepare_train_data(data = data_train_test, hyperparams = hyperparams,
                                    size_sample = size_sample_train_test,
                                    train_ratio = 0.7, seed = seed, verbose = TRUE,
                                    treatment_percentile = 10,binary=binary)

train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
Test_CATT = res_train_test$Test_CATT


## DR-Learner: sequential tuning with separate pi, mu0, mu1, and cate params

# Fit DR-Learner with separate hyperparameters
DR_RF_learner_fit <- function(X, y, t,
                              params_pi, params_mu0, params_mu1, params_cat,
                              seed=42) {
  set.seed(seed)
  idx   <- createDataPartition(y, p=0.5, list=FALSE)
  X_tr  <- X[idx,]; y_tr<-y[idx]; t_tr<-t[idx]
  X_md  <- X[-idx,]; y_md<-y[-idx]; t_md<-t[-idx]

  # propensity
  samp_pi <- max(2, floor(params_pi$sample.fraction * nrow(X_tr)))
  pi_mod <- randomForest(as.factor(t_tr)~., data=X_tr,
                         ntree=params_pi$ntree,
                         mtry=params_pi$mtry,
                         sampsize=samp_pi,
                         nodesize=params_pi$nodesizeSpl)
  pihat <- pmin(pmax(predict(pi_mod, X_md, type="prob")[,2],1e-3),1-1e-3)

  # mu0/mu1
  samp0 <- max(2, floor(params_mu0$sample.fraction * sum(t_tr==0)))
  mu0_mod <- randomForest(y_tr[t_tr==0]~., data=X_tr[t_tr==0,],
                          ntree=params_mu0$ntree,
                          mtry=params_mu0$mtry,
                          sampsize=samp0,
                          nodesize=params_mu0$nodesizeSpl)
  samp1 <- max(2, floor(params_mu1$sample.fraction * sum(t_tr==1)))
  mu1_mod <- randomForest(y_tr[t_tr==1]~., data=X_tr[t_tr==1,],
                          ntree=params_mu1$ntree,
                          mtry=params_mu1$mtry,
                          sampsize=samp1,
                          nodesize=params_mu1$nodesizeSpl)
  mu0hat <- predict(mu0_mod, X_md); mu1hat <- predict(mu1_mod, X_md)

  # DR pseudo
  Z<-t_md; Y<-y_md
  dr_ps <- (Z/pihat - (1-Z)/(1-pihat))*Y +
           ((1-Z/pihat)*mu1hat - (1-(1-Z)/(1-pihat))*mu0hat)

  # cate
  sampc <- max(2, floor(params_cat$sample.fraction * nrow(X_md)))
  cate_mod <- randomForest(dr_ps~., data=X_md,
                            ntree=params_cat$ntree,
                            mtry=params_cat$mtry,
                            sampsize=sampc,
                            nodesize=params_cat$nodesizeAvg)

  list(pi_model=pi_mod,
       mu0_model=mu0_mod,
       mu1_model=mu1_mod,
       cate_model=cate_mod)
}

# Predict DR
DR_RF_learner_predict <- function(fit, newdata) predict(fit$cate_model, newdata)

# Sequential tuning for DR-Learner
DR_RF_learner_optimize <- function(
  train_X, train_y, train_t,
  val_X, val_CATT,
  param_ranges = list(
    mtry            = c(1,2,3,4),
    sample.fraction = c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6),
    nodesizeSpl     = c(1,3,5,10,15,20,25,30),
    nodesizeAvg     = c(1,3,5,10,15,20,25,30),
    ntree           = c(300,500,1000,1500,2000,3000,5000,10000)
  ),
  seed=42, verbose=TRUE
) {
  set.seed(seed)
  default <- list(ntree=1000,
                  mtry=round(ncol(train_X)/2),
                  sample.fraction=0.5,
                  nodesizeSpl=5,
                  nodesizeAvg=5)

  best_pi   <- default
  best_mu0  <- default
  best_mu1  <- default
  best_cat  <- default
  best_pehe <- Inf

  # tune pi
  if(verbose) message("Tuning pi-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_pi; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=trial,
                                       params_mu0=best_mu0,
                                       params_mu1=best_mu1,
                                       params_cat=best_cat, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  pi %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_pi<-trial; if(verbose) message("   -> new best pi, PEHE=",round(pehe,4)) }
  }

  # tune mu0
  if(verbose) message("Tuning mu0-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_mu0; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=best_pi,
                                       params_mu0=trial,
                                       params_mu1=best_mu1,
                                       params_cat=best_cat, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  mu0 %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_mu0<-trial; if(verbose) message("   -> new best mu0, PEHE=",round(pehe,4)) }
  }

  # tune mu1
  if(verbose) message("Tuning mu1-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_mu1; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=best_pi,
                                       params_mu0=best_mu0,
                                       params_mu1=trial,
                                       params_cat=best_cat, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  mu1 %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_mu1<-trial; if(verbose) message("   -> new best mu1, PEHE=",round(pehe,4)) }
  }

  # tune cate
  if(verbose) message("Tuning cate-model...")
  for(param in names(param_ranges)) for(val in param_ranges[[param]]) {
    trial <- best_cat; trial[[param]] <- val
    fit <- tryCatch(DR_RF_learner_fit(train_X, train_y, train_t,
                                       params_pi=best_pi,
                                       params_mu0=best_mu0,
                                       params_mu1=best_mu1,
                                       params_cat=trial, seed=seed), error=function(e) NULL)
    if(is.null(fit)) next
    pehe <- PEHE(val_CATT, DR_RF_learner_predict(fit, val_X))
    if(verbose) message(sprintf("  cate %s=%s -> PEHE=%.4f", param, val, pehe))
    if(pehe<best_pehe){ best_pehe<-pehe; best_cat<-trial; if(verbose) message("   -> new best cate, PEHE=",round(pehe,4)) }
  }

  list(
    best_pi_params   = best_pi,
    best_mu0_params  = best_mu0,
    best_mu1_params  = best_mu1,
    best_cat_params  = best_cat,
    best_pehe        = best_pehe
  )
}
# ─── 1) RA‐Learner ──────────────────────────────────────────────────────────

## RA-Learner: sequential tuning with separate mu0, mu1, and cate params

# Fit RA-Learner with separate hyperparameters
RA_RF_learner_fit <- function(X, y, w,
                              params_mu0, params_mu1, params_cat,
                              seed = 42) {
  set.seed(seed)
  idx  <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr <- X[idx,]; y_tr <- y[idx]; w_tr <- w[idx]
  X_md <- X[-idx,]; y_md <- y[-idx]; w_md <- w[-idx]

  # mu0 model
  samp0 <- max(2, floor(params_mu0$sample.fraction * sum(w_tr==0)))
  mu0_mod <- randomForest(y_tr[w_tr==0] ~ ., data = X_tr[w_tr==0,],
                          ntree    = params_mu0$ntree,
                          mtry     = params_mu0$mtry,
                          sampsize = samp0,
                          nodesize = params_mu0$nodesizeSpl)
  # mu1 model
  samp1 <- max(2, floor(params_mu1$sample.fraction * sum(w_tr==1)))
  mu1_mod <- randomForest(y_tr[w_tr==1] ~ ., data = X_tr[w_tr==1,],
                          ntree    = params_mu1$ntree,
                          mtry     = params_mu1$mtry,
                          sampsize = samp1,
                          nodesize = params_mu1$nodesizeSpl)

  mu0hat <- predict(mu0_mod, X_md)
  mu1hat <- predict(mu1_mod, X_md)

  # pseudo-outcome
  pseu <- w_md * (y_md - mu0hat) + (1 - w_md) * (mu1hat - y_md)

  # cate model
  sampc <- max(2, floor(params_cat$sample.fraction * nrow(X_md)))
  cate_mod <- randomForest(pseu ~ ., data = X_md,
                            ntree    = params_cat$ntree,
                            mtry     = params_cat$mtry,
                            sampsize = sampc,
                            nodesize = params_cat$nodesizeAvg)

  list(mu0_model = mu0_mod,
       mu1_model = mu1_mod,
       cate_model = cate_mod)
}

# Predict
RA_RF_learner_predict <- function(fit, newdata) predict(fit$cate_model, newdata)

# Sequential tuning for RA-Learner
RA_RF_learner_optimize <- function(
  train_X, train_y, train_w,
  val_X, val_CATT,
  param_ranges = list(
    mtry            = c(1,2,3,4),
    sample.fraction = c(0.05,0.1,0.2),
    nodesizeSpl     = c(1,5,10),
    nodesizeAvg     = c(1,5,10),
    ntree           = c(500,1000,2000)
  ),
  seed = 42, verbose = TRUE
) {
  set.seed(seed)

  # defaults
  default <- list(ntree=1000,
                  mtry=round(ncol(train_X)/2),
                  sample.fraction=0.5,
                  nodesizeSpl=5,
                  nodesizeAvg=5)

  best_mu0  <- default
  best_mu1  <- default
  best_cat  <- default
  best_pehe <- Inf

  # tune mu0
  if (verbose) message("Tuning mu0-model...")
  for (param in names(param_ranges)) {
    for (val in param_ranges[[param]]) {
      trial <- best_mu0; trial[[param]] <- val
      fit   <- tryCatch(
        RA_RF_learner_fit(train_X, train_y, train_w,
                           params_mu0 = trial,
                           params_mu1 = best_mu1,
                           params_cat = best_cat, seed=seed),
        error=function(e) NULL
      )
      if (is.null(fit)) next
      preds <- RA_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  mu0 %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_mu0 <- trial
        if (verbose) message("   -> new best mu0, PEHE=", round(pehe,4))
      }
    }
  }

  # tune mu1
  if (verbose) message("Tuning mu1-model...")
  for (param in names(param_ranges)) {
    for (val in param_ranges[[param]]) {
      trial <- best_mu1; trial[[param]] <- val
      fit   <- tryCatch(
        RA_RF_learner_fit(train_X, train_y, train_w,
                           params_mu0 = best_mu0,
                           params_mu1 = trial,
                           params_cat = best_cat, seed=seed),
        error=function(e) NULL
      )
      if (is.null(fit)) next
      preds <- RA_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  mu1 %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_mu1 <- trial
        if (verbose) message("   -> new best mu1, PEHE=", round(pehe,4))
      }
    }
  }

  # tune cate
  if (verbose) message("Tuning cate-model...")
  for (param in names(param_ranges)) {
    for (val in param_ranges[[param]]) {
      trial <- best_cat; trial[[param]] <- val
      fit   <- tryCatch(
        RA_RF_learner_fit(train_X, train_y, train_w,
                           params_mu0 = best_mu0,
                           params_mu1 = best_mu1,
                           params_cat = trial, seed=seed),
        error=function(e) NULL
      )
      if (is.null(fit)) next
      preds <- RA_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  cate %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_cat <- trial
        if (verbose) message("   -> new best cate, PEHE=", round(pehe,4))
      }
    }
  }

  # return all best sets
  list(
    best_mu0_params = best_mu0,
    best_mu1_params = best_mu1,
    best_cat_params = best_cat,
    best_pehe       = best_pehe
  )
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

## R-Learner with RF: sequential hyperparameter tuning

# Fit R-Learner
r_RF_learner_fit <- function(X, y, w,
                              params_m, params_pi, params_cat,
                              seed = 42) {
  set.seed(seed)
  idx   <- createDataPartition(y, p = 0.5, list = FALSE)
  X_tr  <- X[idx,];   y_tr <- y[idx];   w_tr <- w[idx]
  X_md  <- X[-idx,];  y_md <- y[-idx];  w_md <- w[-idx]

  # outcome m(x)
  samp_m <- max(2, floor(params_m$sample.fraction * nrow(X_tr)))
  m_mod <- randomForest(y_tr ~ ., data = X_tr,
                        ntree    = params_m$ntree,
                        mtry     = params_m$mtry,
                        sampsize = samp_m,
                        nodesize = params_m$nodesize)
  m_hat <- predict(m_mod, X_md)

  # propensity pi(x)
  samp_pi <- max(2, floor(params_pi$sample.fraction * nrow(X_tr)))
  pi_mod <- randomForest(as.factor(w_tr) ~ ., data = X_tr,
                         ntree    = params_pi$ntree,
                         mtry     = params_pi$mtry,
                         sampsize = samp_pi,
                         nodesize = params_pi$nodesize)
  pi_hat <- pmin(pmax(predict(pi_mod, X_md, type="prob")[,2],1e-3),1-1e-3)

  # pseudo-outcome
  resid     <- y_md - m_hat
  treat_dev <- w_md - pi_hat
  pseudo    <- resid / treat_dev
  weights   <- treat_dev^2

  # final cate-model
  samp_cat <- max(2, floor(params_cat$sample.fraction * nrow(X_md)))
  cate_mod <- randomForest(pseudo ~ ., data = X_md,
                            ntree    = params_cat$ntree,
                            mtry     = params_cat$mtry,
                            sampsize = samp_cat,
                            nodesize = params_cat$nodesize,
                            weights  = weights)

  list(m_model = m_mod, pi_model = pi_mod, cate_model = cate_mod)
}

# Predict CATE
r_RF_learner_predict <- function(fit, newdata) {
  predict(fit$cate_model, newdata)
}

# Sequential tuning: first m, then pi, then cate
r_RF_learner_optimize <- function(
  train_X, train_y, train_w,
  val_X, val_CATT,
  param_ranges = list(
    mtry            = c(1,2,3,4),
    sample.fraction = c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6),
    nodesize        = c(1,3,5,10,15,20,25,30),
    ntree           = c(300,500,1000,1500,2000,3000,5000,10000)
  ),
  seed = 42, verbose = TRUE
) {
  set.seed(seed)

  # defaults
  best_m  <- list(ntree=1000, mtry=round(ncol(train_X)/2), nodesize=5, sample.fraction=0.5)
  best_pi <- best_m
  best_c  <- best_m
  best_pehe <- Inf

  # tune m-model
  if (verbose) message("Tuning m-model...")
  for (param in names(param_ranges)) {
    if (verbose) message(" optimizing ", param)
    for (val in param_ranges[[param]]) {
      trial <- best_m; trial[[param]] <- val
      fit  <- tryCatch(
        r_RF_learner_fit(train_X, train_y, train_w, trial, best_pi, best_c, seed=seed),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      preds <- r_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_m <- trial
        if (verbose) message("   -> new best m-model, PEHE=", round(pehe,4))
      }
    }
  }

  # tune pi-model
  if (verbose) message("Tuning pi-model...")
  for (param in names(param_ranges)) {
    if (verbose) message(" optimizing ", param)
    for (val in param_ranges[[param]]) {
      trial <- best_pi; trial[[param]] <- val
      fit  <- tryCatch(
        r_RF_learner_fit(train_X, train_y, train_w, best_m, trial, best_c, seed=seed),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      preds <- r_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_pi <- trial
        if (verbose) message("   -> new best pi-model, PEHE=", round(pehe,4))
      }
    }
  }

  # tune cate-model
  if (verbose) message("Tuning cate-model...")
  for (param in names(param_ranges)) {
    if (verbose) message(" optimizing ", param)
    for (val in param_ranges[[param]]) {
      trial <- best_c; trial[[param]] <- val
      fit  <- tryCatch(
        r_RF_learner_fit(train_X, train_y, train_w, best_m, best_pi, trial, seed=seed),
        error = function(e) NULL
      )
      if (is.null(fit)) next
      preds <- r_RF_learner_predict(fit, val_X)
      pehe  <- PEHE(val_CATT, preds)
      if (verbose) message(sprintf("  %s=%s -> PEHE=%.4f", param, val, pehe))
      if (pehe < best_pehe) {
        best_pehe <- pehe; best_c <- trial
        if (verbose) message("   -> new best cate-model, PEHE=", round(pehe,4))
      }
    }
  }

  list(
    best_m_params  = best_m,
    best_pi_params = best_pi,
    best_cat_params= best_c,
    best_pehe      = best_pehe
  )
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

# RLOSS:
start_time <- Sys.time()
best_rloss <- r_RF_learner_optimize(
  train_X   = train_augmX[,-ncol(train_augmX)],
  train_y   = y_train,
  train_w   = z_train,
  val_X     = val_augmX[z_val == 1, ],
  val_CATT  = val_CATT,
param_ranges <- param_ranges,
  seed=42, verbose=TRUE
)
end_time <- Sys.time()
execution_time <- end_time - start_time
print("R-RF opti_time : ")
print(execution_time)
fit_rloss  <- r_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, w = z_train,
      params_m = best_rloss$best_m, params_pi= best_rloss$best_pi,params_cat = best_rloss$best_c )
pred_rloss <- r_RF_learner_predict(fit_rloss, newdata = test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE_RLOSS = PEHE(Test_CATT, pred_rloss[z_test == 1])
print("Rloss - Perf on test data")
print(CATT_Test_PEHE_RLOSS)




# ─── USAGE ─────────────────────────────────────────────────────
# param_ranges <- list(
#  mtry = c(1,2,3,4), sample.fraction = c(0.01,0.05,...),
#  nodesizeSpl = ..., nodesizeAvg = ..., ntree = ...
# )
if (1){
    start_time <- Sys.time()
best_params <- DR_RF_learner_optimize(
  train_X   = train_augmX[,-ncol(train_augmX)],
  train_y   = y_train,
  train_t   = z_train,
  val_X     = val_augmX[z_val == 1, ],
  val_CATT  = val_CATT,
  param_ranges <- param_ranges,
  verbose = TRUE
)
end_time <- Sys.time()
execution_time <- end_time - start_time
print("DR-RF opti_time : ")
print(execution_time)
}
# fit_res     <- dr_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, t = z_train)
fit_res     <- DR_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, t = z_train,
      params_pi=best_params$best_pi, params_mu0=best_params$best_mu0, params_mu1=best_params$best_mu1, params_cat=best_params$best_cat)
cate_preds  <- DR_RF_learner_predict(fit_res, newdata = test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE_DR = PEHE(Test_CATT, cate_preds[z_test == 1])
print("DR - Perf on test data")
print(CATT_Test_PEHE_DR)



# RA:
start_time <- Sys.time()
best_ra <- RA_RF_learner_optimize(
  train_X   = train_augmX[,-ncol(train_augmX)],
  train_y   = y_train,
  train_w   = z_train,
  val_X     = val_augmX[z_val == 1, ],
  val_CATT  = val_CATT,
  param_ranges <- param_ranges,
  verbose = TRUE
)
end_time <- Sys.time()
execution_time <- end_time - start_time
print("RA-RF opti_time : ")
print(execution_time)
fit_ra  <- RA_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, w = z_train,
      params_mu0 =best_ra$best_mu0, params_mu1=best_ra$best_mu1, params_cat=best_ra$best_cat)
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
  param_ranges <- param_ranges,
  verbose = TRUE
)
fit_pw  <- PW_RF_learner_fit(X = train_augmX[, -ncol(train_augmX)], y = y_train, w = z_train, params = best_pw)
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
print("Rloss - Perf on test data")
print(CATT_Test_PEHE_RLOSS)