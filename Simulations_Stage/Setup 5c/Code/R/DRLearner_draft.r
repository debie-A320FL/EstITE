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
source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5c/Code/R/learner_function.R")

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


size_sample_train_test = 1e3
print("size_sample_train_test")
print(size_sample_train_test)
res_train_test = prepare_train_data(data = data_train_test, hyperparams = hyperparams,
                                    size_sample = size_sample_train_test,
                                    train_ratio = 0.7, seed = seed, verbose = TRUE,
                                    treatment_percentile = 10,binary=binary)

train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
Test_CATT = res_train_test$Test_CATT




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