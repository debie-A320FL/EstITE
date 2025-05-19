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

library(tidyverse)
library(causalToolbox)
library(nnet)


source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 4/Code/R/function_RF_optimisation.R")

BScore <- function(x, y) mean((100*x - 100*y)^2)
bias <- function(x, y) mean(100*x - 100*y)
PEHE <- function(x, y) sqrt(mean((100*x - 100*y)^2))

MSE_100 <- function(x, y) sqrt(mean((100*x - 100*y)^2))

MC_se <- function(x, B) qt(0.975, B - 1) * sd(x) / sqrt(B)

r_loss <- function(y, mu, z, pi, tau) mean(((y - mu) - (z - pi) * tau)^2)

# Préparation des données -------------------------------------------------

# Importer les données
# Load Data
curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./../../')

#list_size <- c(500,1000,5000, 10000)


data <- read.csv("./../Setup 1a/Data/simulated_1M_data.csv")

set.seed(123)

data = data[sample(nrow(data)),]
size_sample = 10000
print("size sample")
print(size_sample)
data = data[1:size_sample,]

# Importer les hyperparamètres
hyperparams <- read.csv("./../Setup 1a/Data/hyperparams.csv")

# Extraire les variables nécessaires
myZ <- data$treatment
# myY <- data$Y
myX <- data %>% select(-treatment, -Y) %>% as.matrix()

# Calculer mu_0, tau, et ITE
mu_0 <- hyperparams$gamma_0 + hyperparams$gamma_1 * myX[, "age"] + hyperparams$gamma_2 * myX[, "weight"] + hyperparams$gamma_3 * myX[, "comorbidities"] + hyperparams$gamma_4 * myX[, "gender"]
tau <- hyperparams$delta_0 + hyperparams$delta_1 * myX[, "age"] + hyperparams$delta_2 * myX[, "weight"] + hyperparams$delta_3 * myX[, "comorbidities"] + hyperparams$delta_4 * myX[, "gender"]
ITE <- mu_0 + tau * myZ

# Ajouter une colonne pi pour la probabilité théorique
data$pi <- 1 / (1 + exp(-(mu_0 + tau * myZ)))

ITE_proba <- 1 / (1 + exp(-(mu_0 + tau))) - 1 / (1 + exp(-mu_0))


# Estimation --------------------------------------------------------------

### OPTIONS
N = data %>% nrow()

#### PScore Estimation
## PScore Model - 1 hidden layer neural net
print("estimating PS")
PS_nn <- nnet(x = myX, y = myZ, size = 10, maxit = 2000, 
              decay = 0.01, trace=FALSE, abstol = 1.0e-8) 

PS_est = PS_nn$fitted.values

# Remove unused vars
rm(data)


    bruit_gaussien <- rnorm(size_sample, mean = 0, sd = sqrt(hyperparams$sigma_sq))

    # Appliquer la fonction logistique
    fac = 1
    myY <- plogis(ITE + bruit_gaussien * fac)


    
    
    ### Common mu(x) model for RLOSS evaluation
    #mu_boost = xgboost::xgboost(label = myY, 
     #                           data = myX,           
      #                          max.depth = 3, 
      #                          nround = 300,
      #                          early_stopping_rounds = 4, 
      #                          objective = "reg:squarederror",
      #                          gamma = 1, verbose = F)
   # 
    #mu_est = predict(mu_boost, myX)
    
    
    # Train-Test Splitting
    mysplit <- c(rep(1, ceiling(0.7*N)), 
                 rep(2, floor(0.3*N)))
    
    smp_split <- sample(mysplit, replace = FALSE)  # random permutation
    
    y_train <- myY[smp_split == 1]
    y_test <- myY[smp_split == 2]
    
    x_train <- myX[smp_split == 1, ]
    x_test <- myX[smp_split == 2, ]
    
    z_train <- myZ[smp_split == 1]
    z_test <- myZ[smp_split == 2]
    
    Train_ITE <- ITE_proba[smp_split == 1]
    Test_ITE <- ITE_proba[smp_split == 2]
    
    Train_CATT <- Train_ITE[z_train == 1]; Train_CATC <- Train_ITE[z_train == 0]
    Test_CATT <- Test_ITE[z_test == 1]; Test_CATC <- Test_ITE[z_test == 0]
    
    # Augment X with PS estimates
    train_augmX = cbind(x_train, PS_est[smp_split == 1])
    test_augmX = cbind(x_test, PS_est[smp_split == 2])

if (0){
  # Example usage:
  result <- optimize_and_evaluate_S_RF(
    train_augmX, z_train, y_train, test_augmX, z_test, y_test,
    Train_CATT, verbose=FALSE
  )

  # Access the results:
  best_model <- result$best_model
  best_params <- result$best_params
  print("best param")
  print(best_params)
  best_perf <- result$best_performance
  print("best perf")
  print(best_perf)

  # Extract all performances
  all_performances <- result$all_performances

  # Convert the list of performances to a data frame
  all_performances_df <- do.call(rbind, lapply(all_performances, function(x) {
    data.frame(
      ntree = x$params$ntree,
      mtry = x$params$mtry,
      nodesizeSpl = x$params$nodesizeSpl,
      nodesizeAvg = x$params$nodesizeAvg,
      sample.fraction = x$params$sample.fraction,
      performance = x$performance
    )
  }))

  # Order the data frame by performance
  all_performances_df <- all_performances_df[order(all_performances_df$performance), ]

  # Display the top 20 best performing parameters and their performance
  top_20_performances <- head(all_performances_df, 20)
  print(top_20_performances)
}

if (1){
for (i in 1:1){
cat("\nS_RF\n")
start_time_RF2 <- Sys.time()
result <- optimize_and_evaluate_S_RF_2(
  train_augmX, z_train, y_train, test_augmX, z_test, y_test,
  Test_CATT, verbose=FALSE
)

end_time_RF2 <- Sys.time()
execution_time_RF2 <- end_time_RF2 - start_time_RF2
print("S-RF_execution_time : ")
print(execution_time_RF2)
best_perf <- result$best_performance
print("best perf")
print(best_perf)
best_params <- result$best_params
#print("best param")
#print(best_params)
}
}

if (0){
print("\n\n\n\n")
print("S_RF")
start_time_RF <- Sys.time()
result <- optimize_and_evaluate_S_RF(
  train_augmX, z_train, y_train, test_augmX, z_test, y_test,
  Train_CATT, verbose=FALSE
)

end_time <- Sys.time()
execution_time <- end_time - start_time_RF
print(paste0("S-RF_execution_time : ", execution_time))
best_perf <- result$best_performance
print("best perf")
print(best_perf)
best_params <- result$best_params
#print("best param")
#print(best_params)
}


if (0){
for (i in 1:0){
cat("\nS_RF3_0.95\n")
start_time_RF3 <- Sys.time()
result <- optimize_and_evaluate_S_RF_3(
  train_augmX, z_train, y_train, test_augmX, z_test, y_test,
  Train_CATT, verbose=FALSE, performance_threshold = 0.95
)

end_time_RF3 <- Sys.time()
execution_time_RF3 <- end_time_RF3 - start_time_RF3
print(paste0("S-RF_execution_time : ", execution_time_RF3))
best_perf <- result$best_performance
print("best perf")
print(best_perf)
}


for (i in 1:0){
cat("\nS_RF3_0.90\n")
start_time_RF3 <- Sys.time()
result <- optimize_and_evaluate_S_RF_3(
  train_augmX, z_train, y_train, test_augmX, z_test, y_test,
  Train_CATT, verbose=FALSE, performance_threshold = 0.90
)

end_time_RF3 <- Sys.time()
execution_time_RF3 <- end_time_RF3 - start_time_RF3
print(paste0("S-RF_execution_time : ", execution_time_RF3))
best_perf <- result$best_performance
print("best perf")
print(best_perf)
}
}

cat("\n\n\n\n")
print("T_RF")
start_time <- Sys.time()
result <- optimize_and_evaluate_T_RF(
  train_augmX, z_train, y_train, test_augmX, z_test, y_test,
  Test_CATT, verbose=FALSE
)

end_time <- Sys.time()
execution_time <- end_time - start_time
print("T-RF_execution_time : ")
print(execution_time)
best_perf <- result$best_performance
print("best perf")
print(best_perf)
best_params <- result$best_params

cat("\n\n\n\n")
print("X_RF")
start_time_RF <- Sys.time()
result <- optimize_and_evaluate_X_RF(
  train_augmX, z_train, y_train, test_augmX, z_test, y_test,
  Test_CATT, verbose=FALSE
)

end_time <- Sys.time()
execution_time <- end_time - start_time_RF
print("X-RF_execution_time : ")
print(execution_time)
best_perf <- result$best_performance
print("best perf")
print(best_perf)
best_params <- result$best_params