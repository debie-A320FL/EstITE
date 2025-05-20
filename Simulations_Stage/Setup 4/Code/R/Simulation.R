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

source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 4/Code/R/function_hypparam_optimisation.R")

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

setwd("/home/onyxia/work/EstITE/Simulations_Stage/Setup 4")

hyperparams <- read.csv("./../Setup 1a/Data/hyperparams.csv")

data_train_test <- read.csv("./../Setup 1a/Data/simulated_1M_data.csv")

data_validation <- read.csv("./../Setup 1a/Data/simulated_10K_data_validation.csv")
size_sample_val = nrow(data_validation)
res_val = prepare_train_data(data = data_validation, hyperparams = hyperparams,
                            size_sample = size_sample_val,
                            train_ratio = 0)

val_augmX = res_val$test_augmX; z_val = res_val$z_test; y_val = res_val$y_test
val_CATT = res_val$Test_CATT

list_size <- c(1000)
for (size_sample in list_size) {

  print(paste("size_sample =", size_sample))
  
  # Estimation --------------------------------------------------------------

  ### OPTIONS
  B = 3   # Num of Simulations

  MLearners = c('R-LASSO',"S-RF","T-RF","X-RF",
                  "S-RF-opti", "T-RF-opti", "X-RF-opti",
                  "R-LASSO-opti")

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
    
    cat("\n-------- Iteration", i, "--------\n")
    
    
    if(i<=500){set.seed(502 + i*5); seed = 502 + i*5}
    if(i>500){set.seed(7502 + i*5); seed = 7502 + i*5}

    # Data preparation

    size_sample_train_test = size_sample
    res_train_test = prepare_train_data(data = data_train_test, hyperparams = hyperparams,
                                        size_sample = size_sample_train_test,
                                        train_ratio = 0.7, seed = seed)

    train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
    test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
    Test_CATT = res_train_test$Test_CATT

    
    ###### MODELS ESTIMATION  ------------------------------------------------
    
    
    # ################ S-RF

    start_time <- Sys.time()
    SRF <- S_RF(train_augmX, z_train, y_train)
    train_est = EstimateCate(SRF, train_augmX)
    test_est = EstimateCate(SRF, test_augmX)
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'S-RF'] = as.numeric(execution_time, units = "secs")

    cat("\n\nS-RF_execution_time : ")
    print(execution_time)

    # CATT
    Results$CATT_Test_Bias[i, 'S-RF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'S-RF'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'S-RF'])  

    rm(SRF)

    # ################ S-RF with hyperparameter optimisation
    
    start_time <- Sys.time()
    SRF_result <- optimize_and_evaluate_S_RF_2(
      train_augmX, z_train, y_train, val_augmX, z_val, y_val,
      val_CATT, verbose=FALSE
    )

    end_time <- Sys.time()
    execution_time <- end_time - start_time
    SRF = SRF_result$best_model
    train_est = EstimateCate(SRF, train_augmX)
    test_est = EstimateCate(SRF, test_augmX)
  
    Liste_time$execution_time[i, 'S-RF-opti'] = as.numeric(execution_time, units = "secs")

    cat("\n\nS-RF-opti_execution_time : ")
    print(execution_time)

    # CATT
    Results$CATT_Test_Bias[i, 'S-RF-opti'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'S-RF-opti'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'S-RF-opti']) 

    rm(SRF)
    rm(SRF_result)
    
    
    # #################### T-RF 
    
    start_time <- Sys.time()
    TRF <- T_RF(train_augmX, z_train, y_train)
    train_est = EstimateCate(TRF, train_augmX)
    test_est = EstimateCate(TRF, test_augmX)
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'T-RF'] = as.numeric(execution_time, units = "secs")

    cat("\n\nT-RF_execution_time : ")
    print(execution_time)

    # CATT
    Results$CATT_Test_Bias[i, 'T-RF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'T-RF'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'T-RF']) 

    rm(TRF)

    # #################### T-RF hyper opti
    
    start_time <- Sys.time()
    TRF_result <- optimize_and_evaluate_T_RF(
      train_augmX, z_train, y_train, val_augmX, z_val, y_val,
      val_CATT, verbose=FALSE
    )
    TRF = TRF_result$best_model
    train_est = EstimateCate(TRF, train_augmX)
    test_est = EstimateCate(TRF, test_augmX)
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'T-RF-opti'] = as.numeric(execution_time, units = "secs")

    cat("\n\nT-RF-opti_execution_time : ")
    print(execution_time)

    # CATT
    Results$CATT_Test_Bias[i, 'T-RF-opti'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'T-RF-opti'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'T-RF-opti']) 

    rm(TRF)
    rm(TRF_result)
    
    # #################### X-RF
    # # Remove propensity score
    start_time <- Sys.time()
    XRF <- X_RF(train_augmX[,-ncol(train_augmX)], z_train, y_train)
    train_est = EstimateCate(XRF, train_augmX[,-ncol(train_augmX)])
    test_est = EstimateCate(XRF, test_augmX[,-ncol(test_augmX)])
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'X-RF'] = as.numeric(execution_time, units = "secs")

    cat("\n\nX-RF_execution_time : ")
    print(execution_time)

    # CATT
    Results$CATT_Test_Bias[i, 'X-RF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'X-RF'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'X-RF']) 

    rm(XRF)

    # #################### X-RF hyper opti
    # # Remove propensity score
    start_time <- Sys.time()
    XRF_result <- optimize_and_evaluate_X_RF(
      train_augmX[,-ncol(train_augmX)], z_train, y_train, val_augmX[,-ncol(val_augmX)], z_val, y_val,
      val_CATT, verbose=FALSE
    )
    XRF = XRF_result$best_model
    train_est = EstimateCate(XRF, train_augmX[,-ncol(train_augmX)])
    test_est = EstimateCate(XRF, test_augmX[,-ncol(test_augmX)])
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'X-RF-opti'] = as.numeric(execution_time, units = "secs")

    cat("\n\nX-RF-opti_execution_time : ")
    print(execution_time)

    # CATT
    Results$CATT_Test_Bias[i, 'X-RF-opti'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'X-RF-opti'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'X-RF-opti'])
    
    rm(XRF)
    rm(XRF_result)

    
    
    ######################## R-Lasso Regression
    # No estimated PS as 
    start_time <- Sys.time()
    
    RLASSO <- rlasso(x = train_augmX[, -ncol(train_augmX)], w = z_train, y = y_train,
                     lambda_choice = "lambda.min", rs = FALSE)
    
    train_est = predict(RLASSO, train_augmX[, -ncol(train_augmX)])
    test_est = predict(RLASSO, test_augmX[, -ncol(train_augmX)])
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'R-LASSO'] = as.numeric(execution_time, units = "secs")

    cat("\n\nR-Lasso_execution_time : ")
    print(execution_time)
    
    # CATT 
    Results$CATT_Test_Bias[i, 'R-LASSO'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'R-LASSO'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'R-LASSO'])

    rm(RLASSO)


    ######################## R-Lasso Regression with opti
    # No estimated PS as 
    start_time <- Sys.time()
    
    result <- optimize_and_evaluate_rlasso(
      train_augmX[,-ncol(train_augmX)], z_train, y_train, val_augmX[,-ncol(val_augmX)], z_val, y_val,
      val_CATT, verbose=FALSE
    )
    RLASSO <- result$best_model
    train_est = predict(RLASSO, train_augmX[, -ncol(train_augmX)])
    test_est = predict(RLASSO, test_augmX[, -ncol(train_augmX)])
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'R-LASSO-opti'] = as.numeric(execution_time, units = "secs")

    cat("\n\nR-LASSO-opti_execution_time : ")
    print(execution_time)
    
    # CATT 
    Results$CATT_Test_Bias[i, 'R-LASSO-opti'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'R-LASSO-opti'] = PEHE(Test_CATT, test_est[z_test == 1])
    
    print("Perf on test data")
    print(Results$CATT_Test_PEHE[i, 'R-LASSO-opti'])

    rm(result)
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
                                file=paste0(getwd(), "/Results/Logit_", B, "_", x, "_fac_",fac, "_Nsize_",size_sample,".csv") ) )
 )


 write.csv(sapply( names(Results), function(x) colMeans(Results[[x]]) ), 
           file = paste0(getwd(), "/Results/MeanSummary_", B, "_fac_",fac, "_Nsize_",size_sample,".csv"))
 
 write.csv(sapply( names(Results), function(x) apply(Results[[x]], 2, function(y) MC_se(y, B)) ), 
           file = paste0(getwd(), "/Results/MCSE_Summary_", B, "_fac_",fac, "_Nsize_",size_sample, ".csv"))

 write.csv(Liste_time$execution_time, file = paste0(getwd(), "/Results/Execution time_R_", B, "_fac_",fac, "_Nsize_",size_sample, ".csv"))


}
