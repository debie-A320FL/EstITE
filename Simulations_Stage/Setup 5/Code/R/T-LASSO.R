# Install tidyverse if not already installed
if (!require("tidyverse")) {
  install.packages("tidyverse")
}

# Install causalToolbox from GitHub if not already installed
if (!require("causalToolbox")) {
  devtools::install_github("soerenkuenzel/causalToolbox", upgrade = "never")
}

# Install grf if not already installed
if (!require("nnet")) {
  install.packages("nnet")
}

# Install rlearner from GitHub if not already installed
if (!require("rlearner")) {
  devtools::install_github("xnie/rlearner", upgrade = "never")
}


library(tidyverse)
library(causalToolbox)
library(nnet)
library(rlearner)


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

setwd("/home/onyxia/work/EstITE/Simulations_Stage/Setup 4")

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


size_sample_train_test = 1e5
print("size_sample_train_test")
print(size_sample_train_test)
res_train_test = prepare_train_data(data = data_train_test, hyperparams = hyperparams,
                                    size_sample = size_sample_train_test,
                                    train_ratio = 0.7, seed = seed, verbose = TRUE,
                                    treatment_percentile = 3)

train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
Test_CATT = res_train_test$Test_CATT

#----------------------------------------------
# T_LASSO_train(): Fit two separate Lasso models—
#   one on controls; one on treated
#----------------------------------------------
#
# Inputs:
#   trainX : n×p data.frame or matrix of covariates
#   trainZ : length‐n {0,1} treatment vector
#   trainY : length‐n numeric outcome vector
#
# Outputs:
#   A list with two cv.glmnet objects:
#     $fit_0 for f0(x) (controls),  
#     $fit_1 for f1(x) (treated).
#----------------------------------------------

# Install glmnet if not already installed
if (!require("glmnet")) {
  install.packages("glmnet")
}
library(glmnet)

T_LASSO_train <- function(trainX, trainZ, trainY) {
  X_mat <- as.matrix(trainX)
  Z_vec <- as.numeric(trainZ)
  Y_vec <- as.numeric(trainY)
  
  # Subset for Z=0 (controls)
  idx0 <- which(Z_vec == 0)
  X0   <- X_mat[idx0, , drop = FALSE]
  Y0   <- Y_vec[idx0]
  
  # Subset for Z=1 (treated)
  idx1 <- which(Z_vec == 1)
  X1   <- X_mat[idx1, , drop = FALSE]
  Y1   <- Y_vec[idx1]
  
  # Fit Lasso on controls:
  fit_0 <- cv.glmnet(
    x      = X0,
    y      = Y0,
    family = "gaussian",
    alpha  = 1
  )
  
  # Fit Lasso on treated:
  fit_1 <- cv.glmnet(
    x      = X1,
    y      = Y1,
    family = "gaussian",
    alpha  = 1
  )
  
  return(list(fit_0 = fit_0, fit_1 = fit_1))
}

cat("\n T-LASSO running... \n\n")
start_time <- Sys.time()

# 1) Train the T‐Learner:
out_T <- T_LASSO_train(
  trainX = train_augmX,
  trainZ = z_train,
  trainY = y_train
)
fit_0 <- out_T$fit_0
fit_1 <- out_T$fit_1

# 2) Predict f0 and f1 on the TEST set:
X_test_mat <- as.matrix(test_augmX)
f0_test    <- as.numeric(predict(fit_0, s = "lambda.min", newx = X_test_mat))
f1_test    <- as.numeric(predict(fit_1, s = "lambda.min", newx = X_test_mat))
tau_test_T <- f1_test - f0_test

# 3) Evaluate bias/PEHE on the treated portion of TEST:
tau_test_T_treated <- tau_test_T[z_test == 1]
bias_T  <- bias(Test_CATT, tau_test_T_treated)
PEHE_T  <- PEHE(Test_CATT, tau_test_T_treated)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("\n\n\n")
print(execution_time)

cat("T‐LASSO bias on treated test units:", bias_T, "\n")
cat("T‐LASSO PEHE on treated test units:", PEHE_T, "\n")