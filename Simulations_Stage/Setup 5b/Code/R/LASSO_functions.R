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