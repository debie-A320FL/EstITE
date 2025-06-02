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
# X_LASSO_train(): Fit the two‐stage X‐Learner
#----------------------------------------------
#
# Inputs:
#   trainX : n×p data.frame or matrix of covariates
#   trainZ : length‐n {0,1} treatment vector
#   trainY : length‐n numeric outcome vector
#
# Outputs:
#   A list of four cv.glmnet objects:
#     $f0_fit  = fit for f0(x) on controls,
#     $f1_fit  = fit for f1(x) on treated,
#     $g0_fit  = fit for g0(x) on controls (imputed pseudo‐outcomes),
#     $g1_fit  = fit for g1(x) on treated (imputed pseudo‐outcomes).
#----------------------------------------------

# Install glmnet if not already installed
if (!require("glmnet")) {
  install.packages("glmnet")
}
library(glmnet)

X_LASSO_train <- function(trainX, trainZ, trainY) {
  X_mat <- as.matrix(trainX)
  Z_vec <- as.numeric(trainZ)
  Y_vec <- as.numeric(trainY)
  n     <- nrow(X_mat)
  p     <- ncol(X_mat)
  
  # 1) T‐Learner substep (f0 on controls, f1 on treated)
  idx0 <- which(Z_vec == 0)
  idx1 <- which(Z_vec == 1)
  X0   <- X_mat[idx0, , drop = FALSE]
  Y0   <- Y_vec[idx0]
  X1   <- X_mat[idx1, , drop = FALSE]
  Y1   <- Y_vec[idx1]
  
  f0_fit <- cv.glmnet(
    x      = X0,
    y      = Y0,
    family = "gaussian",
    alpha  = 1
  )
  f1_fit <- cv.glmnet(
    x      = X1,
    y      = Y1,
    family = "gaussian",
    alpha  = 1
  )
  
  # 2) Impute pseudo‐outcomes:
  #    On treated:  D1_i = Y_i – f0(X_i)
  #    On controls: D0_i = f1(X_i) – Y_i
  f0_on_1 <- as.numeric(predict(f0_fit,
                                s      = "lambda.min",
                                newx   = X_mat[idx1, , drop = FALSE]))
  D1_vec  <- Y_vec[idx1] - f0_on_1
  
  f1_on_0 <- as.numeric(predict(f1_fit,
                                s      = "lambda.min",
                                newx   = X_mat[idx0, , drop = FALSE]))
  D0_vec  <- f1_on_0 - Y_vec[idx0]
  
  # 3) Fit g1 on (D1 ~ X) using only treated rows:
  g1_fit <- cv.glmnet(
    x      = X_mat[idx1, , drop = FALSE],
    y      = D1_vec,
    family = "gaussian",
    alpha  = 1
  )
  #    Fit g0 on (D0 ~ X) using only control rows:
  g0_fit <- cv.glmnet(
    x      = X_mat[idx0, , drop = FALSE],
    y      = D0_vec,
    family = "gaussian",
    alpha  = 1
  )
  
  return(list(
    f0_fit = f0_fit,
    f1_fit = f1_fit,
    g0_fit = g0_fit,
    g1_fit = g1_fit
  ))
}

cat("\n X-LASSO running... \n\n")
start_time <- Sys.time()

# 1) Train X-learner
out_X <- X_LASSO_train(trainX = train_augmX, trainZ = z_train, trainY = y_train)
f0_fit <- out_X$f0_fit
f1_fit <- out_X$f1_fit
g0_fit <- out_X$g0_fit
g1_fit <- out_X$g1_fit

# 2) Estimate e(x) using logistic lasso
X_train_mat <- as.matrix(train_augmX)
Z_vec       <- as.numeric(z_train)
Y_vec       <- as.numeric(y_train)

propensity_fit <- cv.glmnet(
  x      = X_train_mat,
  y      = Z_vec,
  family = "binomial",
  alpha  = 1
)

X_test_mat <- as.matrix(test_augmX)
e_x_test <- as.numeric(predict(propensity_fit, newx = X_test_mat, s = "lambda.min", type = "response"))
w_x_test <- 1 - e_x_test

# 3) Predict g0(x), g1(x)
g1_on_test <- as.numeric(predict(g1_fit, s = "lambda.min", newx = X_test_mat))
g0_on_test <- as.numeric(predict(g0_fit, s = "lambda.min", newx = X_test_mat))

# 4) Combine using w(x)
tau_test_X <- w_x_test * g0_on_test + (1 - w_x_test) * g1_on_test

# 5) Evaluate performance
tau_test_X_treated <- tau_test_X[z_test == 1]
bias_X  <- bias(Test_CATT, tau_test_X_treated)
PEHE_X  <- PEHE(Test_CATT, tau_test_X_treated)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("\n\n\n")
print(execution_time)

cat("X‐LASSO bias on treated test units:", bias_X, "\n")
cat("X‐LASSO PEHE on treated test units:", PEHE_X, "\n")