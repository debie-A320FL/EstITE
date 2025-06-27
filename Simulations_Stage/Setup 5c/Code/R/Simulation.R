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

source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5c/Code/R/function_hypparam_optimisation.R")

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

setwd("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5")

hyperparams <- read.csv("./../Setup 1a/Data/hyperparams.csv")

data_train_test <- read.csv("./../Setup 1a/Data/simulated_1M_data.csv")

data_validation <- read.csv("./../Setup 1a/Data/simulated_10K_data_validation.csv")
size_sample_val = nrow(data_validation)

size_sample = 1e4
binary = TRUE # binary value for Y ?



list_treatment_percentile <- c(10)
for (treatment_percentile in list_treatment_percentile) {

  print(paste("treatment_percentile =", treatment_percentile))

  res_val = prepare_train_data(data = data_validation, hyperparams = hyperparams,
                            size_sample = size_sample_val,
                            train_ratio = 0, treatment_percentile = treatment_percentile,
                            verbose = TRUE, binary=binary)

  val_augmX = res_val$test_augmX; z_val = res_val$z_test; y_val = res_val$y_test
  val_CATT = res_val$Test_CATT
  
  # Estimation --------------------------------------------------------------

  ### OPTIONS
  B = 10   # Num of Simulations

  MLearners = c('R-LASSO',"S-RF","T-RF","X-RF")

  nb_learner = length((MLearners))

  mylist = list(
    CATT_Test_Bias = matrix(NA, B, nb_learner),
    CATT_Test_PEHE = matrix(NA, B, nb_learner),
    CATC_Test_Bias = matrix(NA, B, nb_learner),
    CATC_Test_PEHE = matrix(NA, B, nb_learner)
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
                                        train_ratio = 0.7, seed = seed, treatment_percentile = treatment_percentile,
                                        binary=binary)

    train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
    test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
    Test_CATT = res_train_test$Test_CATT; Test_CATC = res_train_test$Test_CATC

    
    ###### MODELS ESTIMATION  ------------------------------------------------
    
    
    # ################ S-RF

    if (i == 1){
      cat("\n\nSearching for best S_RF parameters")
      start_time <- Sys.time()
      SRF_result <- optimize_and_evaluate_S_RF_2(
        train_augmX, z_train, y_train, val_augmX, z_val, y_val,
        val_CATT, verbose=FALSE
      )
      end_time <- Sys.time()
      execution_time <- end_time - start_time
      cat("\n\nS-RF optimisation ended : ")
      print(execution_time)
    }

    start_time <- Sys.time()
    SRF <- S_RF(train_augmX, z_train, y_train,
                mu.forestry = list(
                      relevant.Variable = 1:ncol(train_augmX),
                      ntree = SRF_result$best_params$ntree,
                      replace = TRUE,
                      sample.fraction = SRF_result$best_params$sample.fraction,
                      mtry = SRF_result$best_params$mtry,
                      nodesizeSpl = SRF_result$best_params$nodesizeSpl,
                      nodesizeAvg = SRF_result$best_params$nodesizeAvg,
                      splitratio = 0.5,
                      middleSplit = FALSE
                    ),
                 nthread = 0)


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
    
    cat("Perf CATT on test data : ")
    cat(Results$CATT_Test_PEHE[i, 'S-RF'])

    # CATC
    Results$CATC_Test_Bias[i, 'S-RF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'S-RF'] = PEHE(Test_CATC, test_est[z_test == 0])
    
    cat("Perf CATC on test data : ")
    cat(Results$CATC_Test_PEHE[i, 'S-RF'])   

    rm(SRF)


    # #################### T-RF

    if (i == 1){
      cat("\n\nSearching for best T_RF parameters")
      start_time <- Sys.time()
      TRF_result <- optimize_and_evaluate_T_RF(
        train_augmX, z_train, y_train, val_augmX, z_val, y_val,
        val_CATT, verbose=FALSE
      )
      end_time <- Sys.time()
      execution_time <- end_time - start_time
      cat("\n\nT-RF optimisation ended : ")
      print(execution_time)
    }

    start_time <- Sys.time()
    TRF <- T_RF(train_augmX, z_train, y_train,
                mu0.forestry = list(
                  relevant.Variable = 1:ncol(train_augmX),
                  ntree = TRF_result$best_mu0$ntree,
                  replace = TRUE,
                  sample.fraction = TRF_result$best_mu0$sample.fraction,
                  mtry = TRF_result$best_mu0$mtry,
                  nodesizeSpl = TRF_result$best_mu0$nodesizeSpl,
                  nodesizeAvg = TRF_result$best_mu0$nodesizeAvg,
                  splitratio = 0.5,
                  middleSplit = FALSE
                ),
                mu1.forestry = list(
                  relevant.Variable = 1:ncol(train_augmX),
                  ntree = TRF_result$best_mu1$ntree,
                  replace = TRUE,
                  sample.fraction = TRF_result$best_mu1$sample.fraction,
                  mtry = TRF_result$best_mu1$mtry,
                  nodesizeSpl = TRF_result$best_mu1$nodesizeSpl,
                  nodesizeAvg = TRF_result$best_mu1$nodesizeAvg,
                  splitratio = 0.5,
                  middleSplit = FALSE
                ))
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
    
    cat("Perf CATT on test data : ")
    cat(Results$CATT_Test_PEHE[i, 'T-RF']) 

    # CATC
    Results$CATC_Test_Bias[i, 'T-RF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'T-RF'] = PEHE(Test_CATC, test_est[z_test == 0])
    
    cat("Perf CATC on test data : ")
    cat(Results$CATC_Test_PEHE[i, 'T-RF']) 

    rm(TRF)
    
    # #################### X-RF

    if (i == 1){
      cat("\n\nSearching for best X_RF parameters")
      start_time <- Sys.time()
      XRF_result <- optimize_and_evaluate_X_RF(
        train_augmX[,-ncol(train_augmX)], z_train, y_train, val_augmX[,-ncol(val_augmX)], z_val, y_val,
        val_CATT, verbose=FALSE
      )
      end_time <- Sys.time()
      execution_time <- end_time - start_time
      cat("\n\nX-RF optimisation ended : ")
      print(execution_time)
    }


    # # Remove propensity score
    start_time <- Sys.time()
    XRF <- X_RF(train_augmX[,-ncol(train_augmX)], z_train, y_train,
                mu.forestry = list(
                  relevant.Variable = 1:ncol(train_augmX[,-ncol(train_augmX)]),
                  ntree = XRF_result$best_mu$ntree,
                  replace = TRUE,
                  sample.fraction = XRF_result$best_mu$sample.fraction,
                  mtry = XRF_result$best_mu$mtry,
                  nodesizeSpl = XRF_result$best_mu$nodesizeSpl,
                  nodesizeAvg = XRF_result$best_mu$nodesizeAvg,
                  splitratio = 1,
                  middleSplit = TRUE
                ),
                tau.forestry = list(
                  relevant.Variable = 1:ncol(train_augmX[,-ncol(train_augmX)]),
                  ntree = XRF_result$best_tau$ntree,
                  replace = TRUE,
                  sample.fraction = XRF_result$best_tau$sample.fraction,
                  mtry = XRF_result$best_tau$mtry,
                  nodesizeSpl = XRF_result$best_tau$nodesizeSpl,
                  nodesizeAvg = XRF_result$best_tau$nodesizeAvg,
                  splitratio = 0.8,
                  middleSplit = TRUE
                ),
                e.forestry = list(
                  relevant.Variable = 1:ncol(train_augmX[,-ncol(train_augmX)]),
                  ntree = XRF_result$best_e$ntree,
                  replace = TRUE,
                  sample.fraction = XRF_result$best_e$sample.fraction,
                  mtry = XRF_result$best_e$mtry,
                  nodesizeSpl = XRF_result$best_e$nodesizeSpl,
                  nodesizeAvg = XRF_result$best_e$nodesizeAvg,
                  splitratio = 0.5,
                  middleSplit = FALSE
                ))

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
    
    cat("Perf CATT on test data : ")
    cat(Results$CATT_Test_PEHE[i, 'X-RF'])

    # CATC
    Results$CATC_Test_Bias[i, 'X-RF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'X-RF'] = PEHE(Test_CATC, test_est[z_test == 0])
    
    cat("Perf on CATC test data : ")
    cat(Results$CATC_Test_PEHE[i, 'X-RF']) 

    rm(XRF)

    
    
    ######################## R-Lasso Regression
    
    if (i == 1){
      cat("\nSearching for best rlasso parameters")
      start_time <- Sys.time()
      rlass_result <- optimize_and_evaluate_rlasso(
        train_augmX[,-ncol(train_augmX)], z_train, y_train, val_augmX[,-ncol(val_augmX)], z_val, y_val,
        val_CATT, verbose=FALSE
      )
      end_time <- Sys.time()
      execution_time <- end_time - start_time
      cat("\n\nrlasso optimisation ended : ")
      print(execution_time)
    }

    start_time <- Sys.time()
    
    RLASSO <- rlasso(x = train_augmX[, -ncol(train_augmX)], w = z_train, y = y_train,
                     lambda_choice = "lambda.min", rs = FALSE,
                     alpha = rlass_result$best_alpha)
    
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
    
    cat("Perf on test data : ")
    cat(Results$CATT_Test_PEHE[i, 'R-LASSO'])

    # CATC
    Results$CATC_Test_Bias[i, 'R-LASSO'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'R-LASSO'] = PEHE(Test_CATC, test_est[z_test == 0])
    
    cat("Perf on test data : ")
    cat(Results$CATC_Test_PEHE[i, 'R-LASSO'])

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
