rm(list = ls())

### LIBRARIES
# Install devtools if not already installed
if (!require("devtools")) {
  install.packages("devtools")
}

# Install tidyverse if not already installed
if (!require("tidyverse")) {
  install.packages("tidyverse")
}

# Install SparseBCF from GitHub if not already installed
if (!require("SparseBCF")) {
  devtools::install_github("albicaron/SparseBCF", upgrade = "never")
}

# Install BART if not already installed
if (!require("BART")) {
  install.packages("BART")
}

# Install grf if not already installed
if (!require("grf")) {
  install.packages("grf")
}

# Install rlearner from GitHub if not already installed
if (!require("rlearner")) {
  devtools::install_github("xnie/rlearner", upgrade = "never")
}

# Install causalToolbox from GitHub if not already installed
if (!require("causalToolbox")) {
  devtools::install_github("soerenkuenzel/causalToolbox", upgrade = "never")
}

# Install future if not already installed
if (!require("future")) {
  install.packages("future")
}

# Load the libraries
library(tidyverse)
library(SparseBCF)
library(BART)
library(grf)
library(rlearner)
library(causalToolbox)
library(future)


# Other needed packages

# Install grf if not already installed
if (!require("nnet")) {
  install.packages("nnet")
}

library(nnet)
# library(forestry)

source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5/Code/R/function_hypparam_optimisation.R")

source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5b/Code/R/LASSO_functions.R")

availableCores() # 8 processes in total
plan(multisession)  # use future() to assign and value() function to block subsequent evaluations

### EVALUATION FUNCTIONS
# BScore <- function(x, y) mean((x - y)^2)
# bias <- function(x, y) mean(x - y)
# PEHE <- function(x, y) sqrt(mean((x - y)^2))

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

setwd("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5b")

hyperparams <- read.csv("./../Setup 1a/Data/hyperparams.csv")

data_train_test <- read.csv("./../Setup 1a/Data/simulated_1M_data.csv")

data_validation <- read.csv("./../Setup 1a/Data/simulated_10K_data_validation.csv")
size_sample_val = nrow(data_validation)

size_sample = 1e4

list_treatment_percentile <- c(35,25,15,10,5,3,2,1,0.5,0.3,0.2,0.1)
for (treatment_percentile in list_treatment_percentile) {

  print(paste("treatment_percentile =", treatment_percentile))
  
  # Estimation --------------------------------------------------------------

  ### OPTIONS
  B = 80   # Num of Simulations

  MLearners = c('S-LASSO',"T-LASSO","X-LASSO","R-LASSO", "R-LASSO (paper)")

  nb_learner = length((MLearners))

  mylist = list(
    CATT_Test_Bias = matrix(NA, B, nb_learner),
    CATT_Test_PEHE = matrix(NA, B, nb_learner)
  )

  Results <- map(mylist, `colnames<-`, MLearners)
  rm(mylist)

  mylist = list(
    execution_time = matrix(NA, B, nb_learner)
  )
  Liste_time <- map(mylist, `colnames<-`, MLearners)
  rm(mylist)

  system.time(
  
  
  for (i in 1:B) {
    
    gc()
    
    cat("\n\n***** Iteration", i, " - treatment_percentile : ",treatment_percentile, "*****")
    
    
    if(i<=500){set.seed(502 + i*5); seed = 502 + i*5}
    if(i>500){set.seed(7502 + i*5); seed = 7502 + i*5}

    # Data preparation

    size_sample_train_test = size_sample
    res_train_test = prepare_train_data(data = data_train_test, hyperparams = hyperparams,
                                        size_sample = size_sample_train_test,
                                        train_ratio = 0.7, seed = seed, treatment_percentile = treatment_percentile)

    train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
    test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
    Test_CATT = res_train_test$Test_CATT

    # do not augment X
    train_augmX = train_augmX[, -ncol(train_augmX)]
    test_augmX = test_augmX[, -ncol(test_augmX)]

    
    ###### MODELS ESTIMATION  ------------------------------------------------
    
    
    # ################ S-LASSO

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

    end_time <- Sys.time()
    execution_time <- end_time - start_time
    cat("\n\n")
    print(execution_time)

    Liste_time$execution_time[i, 'S-LASSO'] = as.numeric(execution_time, units = "secs")

    tau_test_S_treated <- tau_test_S[z_test == 1]
    bias_S  <- bias(Test_CATT, tau_test_S_treated)
    PEHE_S  <- PEHE(Test_CATT, tau_test_S_treated)

    cat("S‐LASSO PEHE on treated test units:", PEHE_S, "\n")
    
    Results$CATT_Test_Bias[i, 'S-LASSO'] = bias_S
    Results$CATT_Test_PEHE[i, 'S-LASSO'] = PEHE_S

    rm(fit_S)


    # #################### T-LASSO
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

    end_time <- Sys.time()
    execution_time <- end_time - start_time
    cat("\n\n")
    print(execution_time)
    Liste_time$execution_time[i, 'T-LASSO'] = as.numeric(execution_time, units = "secs")

    # 3) Evaluate bias/PEHE on the treated portion of TEST:
    tau_test_T_treated <- tau_test_T[z_test == 1]
    bias_T  <- bias(Test_CATT, tau_test_T_treated)
    PEHE_T  <- PEHE(Test_CATT, tau_test_T_treated)

    cat("T‐LASSO PEHE on treated test units:", PEHE_T, "\n")
    
    Results$CATT_Test_Bias[i, 'T-LASSO'] = bias_T
    Results$CATT_Test_PEHE[i, 'T-LASSO'] = PEHE_T
    
    rm(out_T)

    # #################### X-RF

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

    end_time <- Sys.time()
    execution_time <- end_time - start_time
    cat("\n\n")
    print(execution_time)
    Liste_time$execution_time[i, 'X-LASSO'] = as.numeric(execution_time, units = "secs")


    # 5) Evaluate performance
    tau_test_X_treated <- tau_test_X[z_test == 1]
    bias_X  <- bias(Test_CATT, tau_test_X_treated)
    PEHE_X  <- PEHE(Test_CATT, tau_test_X_treated)

    cat("X‐LASSO PEHE on treated test units:", PEHE_X, "\n")
    
    Results$CATT_Test_Bias[i, 'X-LASSO'] = bias_X
    Results$CATT_Test_PEHE[i, 'X-LASSO'] = PEHE_X

    rm(out_X,propensity_fit)

    
    
    ######################## R-Lasso Regression
    
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

    

    end_time <- Sys.time()
    execution_time <- end_time - start_time
    cat("\n\n")
    print(execution_time)
    Liste_time$execution_time[i, 'R-LASSO'] = as.numeric(execution_time, units = "secs")

    tau_test_R_treated <- tau_test_R[z_test == 1]
    bias_R  <- bias(Test_CATT, tau_test_R_treated)
    PEHE_R  <- PEHE(Test_CATT, tau_test_R_treated)

    cat("R‐LASSO PEHE on treated test units:", PEHE_R, "\n")
    
    Results$CATT_Test_Bias[i, 'R-LASSO'] = bias_R
    Results$CATT_Test_PEHE[i, 'R-LASSO'] = PEHE_R

    rm(out_R)

    ######################## R-Lasso Regression (paper)

    start_time <- Sys.time()
    
    RLASSO <- rlasso(x = train_augmX, w = z_train, y = y_train,
                     lambda_choice = "lambda.min", rs = FALSE)
    
    train_est = predict(RLASSO, train_augmX)
    test_est = predict(RLASSO, test_augmX)
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'R-LASSO (paper)'] = as.numeric(execution_time, units = "secs")

    cat("\n\n")
    print(execution_time)
    
    # CATT 
    Results$CATT_Test_Bias[i, 'R-LASSO (paper)'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'R-LASSO (paper)'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    PEHE_R_paper = Results$CATT_Test_PEHE[i, 'R-LASSO (paper)']
    cat("R‐LASSO (paper) PEHE on treated test units:", PEHE_R_paper, "\n")
    
    rm(RLASSO)


  }
  
)

print("\nSample_size\n")
print(size_sample)

print("\nMean_results\n")
print(sapply( names(Results), function(x) colMeans(Results[[x]]) ))
#sapply( names(Results), function(x) apply(Results[[x]], 2, function(y) MC_se(y, B)) )

print("\nMean_time\n")
print(sapply( names(Liste_time), function(x) colMeans(Liste_time[[x]]) ))

#print("\nSD_time\n")
#print(sapply( names(Liste_time), function(x) apply(Liste_time[[x]], 2, function(y) MC_se(y, B)) ))


cat("\n\n\n")


# Save Results --------------------------------------------------
fac = 1
directory_path <- "./Results"

if (!dir.exists(directory_path)) {
  dir.create(directory_path, recursive = TRUE)
}

 invisible(
   sapply(names(Results), 
          function(x) write.csv(Results[[x]], 
                                file=paste0(getwd(), "/Results/Logit_", B, "_", x, "_treat_p_",treatment_percentile, "_Nsize_",size_sample,".csv") ) )
 )


 write.csv(sapply( names(Results), function(x) colMeans(Results[[x]]) ), 
           file = paste0(getwd(), "/Results/MeanSummary_", B, "_treat_p_",treatment_percentile, "_Nsize_",size_sample,".csv"))
 
 write.csv(sapply( names(Results), function(x) apply(Results[[x]], 2, function(y) MC_se(y, B)) ), 
           file = paste0(getwd(), "/Results/MCSE_Summary_", B, "_treat_p_",treatment_percentile, "_Nsize_",size_sample, ".csv"))

 write.csv(Liste_time$execution_time, file = paste0(getwd(), "/Results/Execution time_R_", B, "_treat_p_",treatment_percentile, "_Nsize_",size_sample, ".csv"))


}
