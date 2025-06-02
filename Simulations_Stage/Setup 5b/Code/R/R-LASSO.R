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
# R_LASSO_train(): R‐Learner with Lasso
#----------------------------------------------
#
# This function fits the R‐Learner using glmnet:
#  1) π̂(x) via logistic Lasso on (trainX → trainZ)
#  2) m̂(x) via linear Lasso on (trainX → trainY)
#  3) Compute residuals R_i = Y_i − m̂(X_i), W_i = Z_i − π̂(X_i)
#  4) Fit τ‐model by regressing R_i on {W_i * X_{ij}} via linear Lasso (no intercept).
#
# Inputs:
#   trainX  : n×p data.frame or matrix of covariates
#   trainZ  : length‐n {0,1} treatment vector
#   trainY  : length‐n numeric outcome vector
#
# Output (list):
#   $pi_fit    : cv.glmnet object (logistic) for π̂(x)
#   $m_fit     : cv.glmnet object (linear) for m̂(x)
#   $tau_fit   : cv.glmnet object (no‐intercept) for τ on features {W*X}
#
# Example usage (see below) shows how to predict τ̂ on a test set and compute bias/PEHE.
#----------------------------------------------

# Install/load glmnet if needed
if (!require("glmnet")) {
  install.packages("glmnet")
}
library(glmnet)

R_LASSO_train <- function(trainX, trainZ, trainY) {
  X_mat <- as.matrix(trainX)           # n × p
  Z_vec <- as.numeric(trainZ)          # length‐n
  Y_vec <- as.numeric(trainY)          # length‐n
  p     <- ncol(X_mat)
  
  # (1) Fit π̂(x) via logistic Lasso on TRAIN:
  pi_fit <- cv.glmnet(
    x      = X_mat,
    y      = Z_vec,
    family = "binomial",
    alpha  = 1
  )
  # π̂ on train (OOB not directly used; we’ll refit on full data for test if needed):
  pi_hat_train <- as.numeric(
    predict(pi_fit, s = "lambda.min", newx = X_mat, type = "response")
  )
  
  # (2) Fit m̂(x) via linear Lasso on TRAIN:
  m_fit <- cv.glmnet(
    x      = X_mat,
    y      = Y_vec,
    family = "gaussian",
    alpha  = 1
  )
  m_hat_train <- as.numeric(
    predict(m_fit, s = "lambda.min", newx = X_mat)
  )
  
  # (3) Compute pseudo‐outcomes/residuals on TRAIN:
  R_vec <- Y_vec - m_hat_train
  W_vec <- Z_vec - pi_hat_train
  
  # (4) Build R‐design: each column j = W_i * X_{ij}
  R_design <- X_mat * W_vec   # n×p
  
  # (5) Fit τ‐model: R_vec ~ R_design (no intercept)
  tau_fit <- cv.glmnet(
    x         = R_design,
    y         = R_vec,
    family    = "gaussian",
    alpha     = 1,
    intercept = FALSE
  )
  
  return(list(
    pi_fit  = pi_fit,
    m_fit   = m_fit,
    tau_fit = tau_fit
  ))
}

#----------------------------------------------
# Example usage: training R‐Learner and evaluating on TEST
#----------------------------------------------

# Assumes in your workspace:
#   train_augmX  : n×p data.frame or matrix of covariates (TRAIN)
#   z_train      : length‐n {0,1} treatment vector (TRAIN)
#   y_train      : length‐n numeric outcome vector (TRAIN)
#   test_augmX   : m×p data.frame or matrix of covariates (TEST)
#   z_test       : length‐m {0,1} treatment vector (TEST)
#   Test_CATT    : length = sum(z_test == 1) “ground truth” CATT on treated TEST units

cat("\nR‐LASSO training and evaluation...\n\n")
start_time <- Sys.time()

# 1) Train the R‐Learner:
out_R <- R_LASSO_train(
  trainX = train_augmX,
  trainZ = z_train,
  trainY = y_train
)
pi_fit  <- out_R$pi_fit
m_fit   <- out_R$m_fit
tau_fit <- out_R$tau_fit

# 2) Recompute π̂_train and m̂_train to get R_vec, W_vec (for tau_train if desired):
X_train_mat   <- as.matrix(train_augmX)
Z_vec         <- as.numeric(z_train)
Y_vec         <- as.numeric(y_train)

pi_hat_train <- as.numeric(
  predict(pi_fit, s = "lambda.min", newx = X_train_mat, type = "response")
)
m_hat_train  <- as.numeric(
  predict(m_fit, s = "lambda.min", newx = X_train_mat)
)

R_vec <- Y_vec - m_hat_train
W_vec <- Z_vec - pi_hat_train

# 3) Extract β_R (length‐p) from tau_fit:
coef_tau   <- as.vector(coef(tau_fit, s = "lambda.min"))
# Because intercept=FALSE, coef_tau[1]=0, coef_tau[2:(p+1)] = β_R
beta_R     <- coef_tau[-1]

# 4) Compute τ̂ on TRAIN if desired:
tau_train_R <- as.numeric(X_train_mat %*% beta_R)

# 5) Compute τ̂ on TEST:
X_test_mat  <- as.matrix(test_augmX)
tau_test_R  <- as.numeric(X_test_mat %*% beta_R)

# 6) Evaluate on the treated portion of TEST:
tau_test_R_treated <- tau_test_R[z_test == 1]
bias_R  <- bias(Test_CATT, tau_test_R_treated)
PEHE_R  <- PEHE(Test_CATT, tau_test_R_treated)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("\n\nTraining & evaluation time:\n")
print(execution_time)

cat("R‐LASSO bias on treated test units:", bias_R, "\n")
cat("R‐LASSO PEHE on treated test units:", PEHE_R, "\n")
