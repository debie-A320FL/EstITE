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
# S_LASSO_train(): Fit one Lasso on [X, Z, Z*X]
#----------------------------------------------
#
# Inputs:
#   trainX : n×p data.frame or matrix of covariates
#   trainZ : length‐n {0,1} treatment vector
#   trainY : length‐n numeric outcome vector
#
# Outputs:
#   A single cv.glmnet object (family="gaussian", alpha=1),
#   trained on the design [X | Z | (Z*X)].
#----------------------------------------------

# Install glmnet if not already installed
if (!require("glmnet")) {
  install.packages("glmnet")
}

library(glmnet)

S_LASSO_train <- function(trainX, trainZ, trainY) {
  X_mat <- as.matrix(trainX)
  n     <- nrow(X_mat)
  p     <- ncol(X_mat)
  Z_vec <- as.numeric(trainZ)
  
  # Build interaction matrix: each column_j = trainX[,j] * Z_vec
  ZX_mat <- X_mat * Z_vec
  
  # Final design: [ X | Z | ZX ]
  design_mat <- cbind(X_mat, Z = Z_vec, ZX_mat)
  
  # Fit Lasso via cv.glmnet
  fit_s <- cv.glmnet(
    x      = design_mat,
    y      = trainY,
    family = "gaussian",
    alpha  = 1
  )
  return(fit_s)
}

cat("\n S-LASSO runing... \n\n")
start_time <- Sys.time()
# 1) Train the S‐Learner:
fit_S <- S_LASSO_train(
  trainX = train_augmX,
  trainZ = z_train,
  trainY = y_train
)

# 2) Grab the coefficient vector at lambda.min:
coef_s   <- as.vector(coef(fit_S, s = "lambda.min"))
# glmnet’s coef() returns length = 1 + (# columns in design_mat).  Namely:
#   coef_s[1]           = intercept
#   coef_s[2:(p+1)]     = β_X      (for each of the p columns of trainX)
#   coef_s[p+2]         = β_Z      (coefficient on the “Z” column)
#   coef_s[(p+3):(2p+2)] = β_{ZX}  (p coefficients on ZX_mat)

p       <- ncol(train_augmX)
beta_Z  <- coef_s[p + 2]
beta_ZX <- coef_s[(p + 3):(2*p + 2)]

# 4) Compute τ̂ on the TEST set:
X_test_mat <- as.matrix(test_augmX)
tau_test_S <- as.numeric(beta_Z + (X_test_mat %*% beta_ZX))

# 5) Evaluate bias/PEHE on the treated portion of TEST:
tau_test_S_treated <- tau_test_S[z_test == 1]
bias_S  <- bias(Test_CATT, tau_test_S_treated)
PEHE_S  <- PEHE(Test_CATT, tau_test_S_treated)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("\n\n\n")
print(execution_time)

cat("S‐LASSO bias on treated test units:", bias_S, "\n")
cat("S‐LASSO PEHE on treated test units:", PEHE_S, "\n")
