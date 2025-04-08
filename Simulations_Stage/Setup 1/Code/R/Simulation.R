##############################
# Simulation using ACTG data #
##############################

rm(list = ls())

### LIBRARIES
library(tidyverse)
library(SparseBCF) # BCF package: install from https://github.com/albicaron/SparseBCF
library(BART) # Main package including all the version of BART
library(grf)
library(rlearner) # from https://github.com/xnie/rlearner
library(causalToolbox) # from https://github.com/soerenkuenzel/causalToolbox
library(future)

# Other needed packages
library(nnet)
# library(forestry)

availableCores() # 8 processes in total
plan(multisession)  # use future() to assign and value() function to block subsequent evaluations

### EVALUATION FUNCTIONS
BScore <- function(x, y) mean((x - y)^2)
bias <- function(x, y) mean(x - y)
PEHE <- function(x, y) sqrt(mean((x - y)^2))
MC_se <- function(x, B) qt(0.975, B - 1) * sd(x) / sqrt(B)

r_loss <- function(y, mu, z, pi, tau) mean(((y - mu) - (z - pi) * tau)^2)

### OPTIONS
B = 3   # Num of Simulations

# Importer les données
# Load Data
curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./../../Data')

data <- read.csv("simulated_data.csv")

# Importer les hyperparamètres
hyperparams <- read.csv("hyperparams.csv")

# Extraire les variables nécessaires
myZ <- data$treatment
myY <- data$Y
myX <- data %>% select(-treatment, -Y) %>% as.matrix()

# Calculer mu_0, tau, et ITE
mu_0 <- hyperparams$gamma_0 + hyperparams$gamma_1 * myX[, "age"] + hyperparams$gamma_2 * myX[, "weight"] + hyperparams$gamma_3 * myX[, "comorbidities"] + hyperparams$gamma_4 * myX[, "gender"]
tau <- hyperparams$delta_0 + hyperparams$delta_1 * myX[, "age"] + hyperparams$delta_2 * myX[, "weight"] + hyperparams$delta_3 * myX[, "comorbidities"] + hyperparams$delta_4 * myX[, "gender"]
ITE <- mu_0 + tau * myZ

# Remove unused vars
rm(data, hyperparams)

# Store simulations true and estimated quantities for CATT and CATC separately
MLearners = c('S-BART', 'T-BART', 'X-BART',
              'R-LASSO', 'R-BOOST', 'CF', 'BCF')

nb_learner = length(MLearners)

mylist = list(
  CATT_Train_Bias = matrix(NA, B, nb_learner),  CATT_Test_Bias = matrix(NA, B, nb_learner),
  CATT_Train_PEHE = matrix(NA, B, nb_learner),  CATT_Test_PEHE = matrix(NA, B, nb_learner),
  CATT_Train_RLOSS = matrix(NA, B, nb_learner), CATT_Test_RLOSS = matrix(NA, B, nb_learner),
  CATC_Train_Bias = matrix(NA, B, nb_learner),  CATC_Test_Bias = matrix(NA, B, nb_learner),
  CATC_Train_PEHE = matrix(NA, B, nb_learner),  CATC_Test_PEHE = matrix(NA, B, nb_learner),
  CATC_Train_RLOSS = matrix(NA, B, nb_learner), CATC_Test_RLOSS = matrix(NA, B, nb_learner)
)

Results <- map(mylist, `colnames<-`, MLearners)
rm(mylist)

mylist = list(
  execution_time = matrix(NA, B, nb_learner)
)
Liste_time <- map(mylist, `colnames<-`, MLearners)

system.time(
  for (i in 1:B) {
    gc()
    
    cat("\n\n\n\n-------- Iteration", i, "--------\n\n\n\n")
    
    if (i <= 500) {
      set.seed(502 + i * 5)
    }
    if (i > 500) {
      set.seed(7502 + i * 5)
    }
    
    # Train-Test Splitting
    mysplit <- c(rep(1, ceiling(0.7 * N)),
                 rep(2, floor(0.3 * N)))
    
    smp_split <- sample(mysplit, replace = FALSE)  # random permutation
    
    y_train <- myY[smp_split == 1]
    y_test <- myY[smp_split == 2]
    
    x_train <- myX[smp_split == 1, ]
    x_test <- myX[smp_split == 2, ]
    
    z_train <- myZ[smp_split == 1]
    z_test <- myZ[smp_split == 2]
    
    Train_ITE <- ITE[smp_split == 1]
    Test_ITE <- ITE[smp_split == 2]
    
    Train_CATT <- Train_ITE[z_train == 1]
    Train_CATC <- Train_ITE[z_train == 0]
    Test_CATT <- Test_ITE[z_test == 1]
    Test_CATC <- Test_ITE[z_test == 0]
    
    # Augment X with treatment estimates
    train_augmX <- cbind(x_train, z_train)
    test_augmX <- cbind(x_test, z_test)
    
    ###### MODELS ESTIMATION  ------------------------------------------------
    
    ######################### S-BART
    #### Train
    start_time <- Sys.time()
    
    myBART <- wbart(x.train = train_augmX, y.train = y_train,
                    x.test = test_augmX, nskip = 2000, ndpost = 4000, printevery = 6000)
    
    XZ0_train <- cbind(train_augmX, z_train)
    XZ0_train[, "z_train"] <- ifelse(XZ0_train[, "z_train"] == 1, 0, 1)
    
    Y0_train <- predict(myBART, newdata = XZ0_train)
    
    All_obs <- cbind(Y1 = myBART$yhat.train.mean,
                     train_augmX,
                     z_train)
    All_count <- cbind(Y0 = colMeans(Y0_train),
                       XZ0_train)
    
    All_Trt <- All_obs
    All_Trt[which(All_Trt[, "z_train"] == 0), ] <- All_count[which(All_count[, "z_train"] == 1), ]
    
    All_Ctrl <- All_count
    All_Ctrl[which(All_Ctrl[, "z_train"] == 1), ] <- All_obs[which(All_obs[, "z_train"] == 0), ]
    
    train_est = All_Trt[, "Y1"] - All_Ctrl[, "Y0"]
    
    #### Test
    XZ0_test <- cbind(test_augmX, z_test)
    XZ0_test[, "z_test"] <- ifelse(XZ0_test[, "z_test"] == 1, 0, 1)
    
    Y0_test <- predict(myBART, newdata = XZ0_test)
    
    All_obs <- cbind(Y1 = myBART$yhat.test.mean,
                     test_augmX,
                     z_test)
    All_count <- cbind(Y0 = colMeans(Y0_test),
                       XZ0_test)
    
    All_Trt <- All_obs
    All_Trt[which(All_Trt[, "z_test"] == 0), ] <- All_count[which(All_count[, "z_test"] == 1), ]
    
    All_Ctrl <- All_count
    All_Ctrl[which(All_Ctrl[, "z_test"] == 1), ] <- All_obs[which(All_obs[, "z_test"] == 0), ]
    
    test_est = All_Trt[, "Y1"] - All_Ctrl[, "Y0"]
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'S-BART'] = execution_time
    
    # CATT
    Results$CATT_Train_Bias[i, 'S-BART'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'S-BART'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'S-BART'] = r_loss(y_train[z_train == 1], mu_0[smp_split == 1][z_train == 1],
                                                   z_train[z_train == 1], myZ[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'S-BART'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'S-BART'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'S-BART'] = r_loss(y_test[z_test == 1], mu_0[smp_split == 2][z_test == 1],
                                                  z_test[z_test == 1], myZ[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'S-BART'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'S-BART'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'S-BART'] = r_loss(y_train[z_train == 0], mu_0[smp_split == 1][z_train == 0],
                                                   z_train[z_train == 0], myZ[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'S-BART'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'S-BART'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'S-BART'] = r_loss(y_test[z_test == 0], mu_0[smp_split == 2][z_test == 0],
                                                  z_test[z_test == 0], myZ[smp_split == 2][z_test == 0], test_est[z_test == 0])
    
    # Free up space
    rm(myBART, Y0_train, Y0_test)
    
    # suite du code avec les autres méthodes...
  }
)
