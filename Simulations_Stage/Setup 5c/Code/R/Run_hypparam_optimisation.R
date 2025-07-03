# Install tidyverse if not already installed
if (!require("tidyverse")) {
  install.packages("tidyverse")
}

# Install causalToolbox from GitHub if not already installed
if (!require("causalToolbox")) {
  devtools::install_github("debie-A320FL/causalToolbox", upgrade = "never")
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


size_sample_train_test = 1000
print("size_sample_train_test")
print(size_sample_train_test)
res_train_test = prepare_train_data(data = data_train_test, hyperparams = hyperparams,
                                    size_sample = size_sample_train_test,
                                    train_ratio = 0.7, seed = seed, verbose = TRUE)

train_augmX = res_train_test$train_augmX; z_train = res_train_test$z_train; y_train = res_train_test$y_train
test_augmX = res_train_test$test_augmX; z_test = res_train_test$z_test; y_test = res_train_test$y_test
Test_CATT = res_train_test$Test_CATT


if (0){
cat("\n\n\n\n")
print("rlasso without opti")
start_time <- Sys.time()
model <- rlasso(x = train_augmX[, -ncol(train_augmX)], w = z_train, y = y_train,
                     lambda_choice = "lambda.min", rs = FALSE)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("rlasso_execution_time : ")
print(execution_time)
test_est = predict(model, test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
print("Perf on test data")
print(CATT_Test_PEHE)
}


if (0){
cat("\n\n\n\n")
print("rlasso")
start_time <- Sys.time()
result <- optimize_and_evaluate_rlasso(
  train_augmX[,-ncol(train_augmX)], z_train, y_train, val_augmX[,-ncol(val_augmX)], z_val, y_val,
  val_CATT, verbose=FALSE
)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("rlasso_execution_time : ")
print(execution_time)
best_perf <- result$best_performance
print("best perf on validation")
print(best_perf)
model <- result$best_model
test_est = predict(model, test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
print("Perf on test data")
print(CATT_Test_PEHE)
best_alpha <- result$best_alpha
print("best alpha")
print(best_alpha)


param_grid <- expand.grid(
    sample.fraction =c(0.05,0.1, 0.15, 0.2),  # Fraction of samples used for each tree
    ntree = c(1000, 3000, 5000),  # Number of trees
    mtry = c(2),  # Features to try at each split
    nodesizeSpl = c(10, 15,20, 25, 30, 35),  # Minimum node size for splits
    nodesizeAvg = c(10, 15,20, 25, 30, 35)  # Minimum node size for averages
  )

cat("\nS_RF (hand)\n")
start_time_RF2 <- Sys.time()
result <- optimize_and_evaluate_S_RF(
  train_augmX, z_train, y_train, val_augmX, z_val, y_val,
  val_CATT, verbose=TRUE, param_grid=param_grid
)

end_time_RF2 <- Sys.time()
execution_time_RF2 <- end_time_RF2 - start_time_RF2
print("S-RF_execution_time : ")
print(execution_time_RF2)

all_perf <- result$all_performances
print(all_perf)

best_perf <- result$best_performance
print("best perf on validation")
print(best_perf)
model <- result$best_model
test_est = EstimateCate(model, test_augmX)
CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
print("Perf on test data")
print(CATT_Test_PEHE)
}

cat("\nS_RF\n")
start_time_RF2 <- Sys.time()
result <- optimize_and_evaluate_S_RF_2(
  train_augmX, z_train, y_train, val_augmX, z_val, y_val,
  val_CATT, verbose=FALSE
)

end_time_RF2 <- Sys.time()
execution_time_RF2 <- end_time_RF2 - start_time_RF2
print("S-RF_execution_time : ")
print(execution_time_RF2)
best_perf <- result$best_performance
print("best perf on validation")
print(best_perf)
model <- result$best_model
test_est = EstimateCate(model, test_augmX)
CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
print("Perf on test data")
print(CATT_Test_PEHE)

print(result$best_params)

cat("\n\n\n\n")
print("T_RF")
start_time <- Sys.time()
result <- optimize_and_evaluate_T_RF(
  train_augmX, z_train, y_train, val_augmX, z_val, y_val,
  val_CATT, verbose=FALSE
)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("T-RF_execution_time : ")
print(execution_time)
best_perf <- result$best_performance
print("best perf on validation")
print(best_perf)
model <- result$best_model
test_est = EstimateCate(model, test_augmX)
CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
print("Perf on test data")
print(CATT_Test_PEHE)


cat("\n\n\n\n")
print("X_RF")
start_time_RF <- Sys.time()
result <- optimize_and_evaluate_X_RF(
  train_augmX[,-ncol(train_augmX)], z_train, y_train, val_augmX[,-ncol(val_augmX)], z_val, y_val,
  val_CATT, verbose=FALSE
)

end_time <- Sys.time()
execution_time <- end_time - start_time_RF
print("X-RF_execution_time : ")
print(execution_time)
best_perf <- result$best_performance
print("best perf on validation")
print(best_perf)
model <- result$best_model
test_est = EstimateCate(model, test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
print("Perf on test data")
print(CATT_Test_PEHE)


cat("\n\n\n\n")
print("X_logit_RF")
start_time_RF <- Sys.time()
result <- optimize_and_evaluate_X_logit_RF(
  train_augmX[,-ncol(train_augmX)], z_train, y_train, val_augmX[,-ncol(val_augmX)], z_val, y_val,
  val_CATT, verbose=FALSE
)

end_time <- Sys.time()
execution_time <- end_time - start_time_RF
print("X-logit_RF_execution_time : ")
print(execution_time)
best_perf <- result$best_performance
print("best perf on validation")
print(best_perf)
model <- result$best_model
test_est = EstimateCate(model, test_augmX[,-ncol(test_augmX)])
CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
print("Perf on test data")
print(CATT_Test_PEHE)
