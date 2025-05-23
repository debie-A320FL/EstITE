##############################
# Simulation using ACTG data #
##############################

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
curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./../../')

data <- read.csv("./../Setup 3/Data/simulated_1M_data_unbalanced_set.csv")

data_b <- data

set.seed(123)
data = data[sample(nrow(data)),]
sample_size = 1000
data = data[1:sample_size,]

# Importer les hyperparamètres
hyperparams <- read.csv("./../Setup 3/Data/hyperparams.csv")

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




# Charger la bibliothèque nécessaire
library(stats)
 
# S_logit <- function(data){
#   # Créer la formule avec les interactions spécifiques
#   formula <- Y ~ age + weight + comorbidities + gender + treatment
#   
#   # Ajuster le modèle de régression logistique
#   model <- glm(formula, data = data, family = binomial(link = "logit"))
#   
#   return(model)
# }
# 
# res_S <- S_logit(data)
# summary(res_S)
# 
# 
# T_logit <- function(data){
#   # Créer la formule avec les interactions spécifiques
#   formula <- Y ~ age + weight + comorbidities + gender
#   
#   # Ajuster le modèle de régression logistique
#   model0 <- glm(formula, data = data %>% filter(treatment == 0), family = binomial(link = "logit"))
#   model1 <- glm(formula, data = data %>% filter(treatment == 1), family = binomial(link = "logit"))
#   
#   return(list(model0 = model0, model1 = model1))
# }
# 
# res_T <- T_logit(data)
# summary(res_T$model0)
# summary(res_T$model1)

S_logit_int <- function(data){
  # Créer la formule avec les interactions spécifiques
  formula <- Y ~ age + weight + comorbidities + gender + treatment +
    treatment:age + treatment:weight + treatment:comorbidities + treatment:gender
  
  # Ajuster le modèle de régression logistique
  model <- glm(formula, data = data, family = binomial(link = "logit"))
  
  return(model)
}

res_S_int <- S_logit_int(data)
summary(res_S_int)


# Prédire les probabilités que Y soit égal à 1
logit_model <- res_S_int
data$prob_logit_Y_1 <- predict(logit_model, type = "response")

MSE_100(data$pi,data$prob_logit_Y_1)

# Création du plot de la MSE du la prédiction issu du logit
# en fonction de la taille de l'échantillon

if (0){
sample_size <- round(logseq(100, nrow(data)))


vec_MSE = c()
#for (size in sample_size){
for (i in 1:length(sample_size)){
  size = sample_size[i]
  print(i)
  data_reduced = data_b[sample(nrow(data)),]
  data_reduced = data_reduced[1:size,]
  
  res_S_int <- S_logit_int(data_reduced)
  # Prédire les probabilités que Y soit égal à 1
  logit_model <- res_S_int
  data_reduced$prob_logit_Y_1 <- NA
  data_reduced$prob_logit_Y_1 <- predict(logit_model, type = "response")
  
  MSE_value = MSE_100(data_reduced$pi,data_reduced$prob_logit_Y_1)
  vec_MSE = c(vec_MSE, MSE_value)
}

plot(sample_size, vec_MSE, log="xy")
}





# Estimation --------------------------------------------------------------

### OPTIONS
B = 80   # Num of Simulations
N = data %>% nrow()

#### PScore Estimation
## PScore Model - 1 hidden layer neural net
PS_nn <- nnet(x = myX, y = myZ, size = 10, maxit = 2000, 
              decay = 0.01, trace=FALSE, abstol = 1.0e-8) 

PS_est = PS_nn$fitted.values

# Remove unused vars
rm(data)

# MLearner without RF one (which do not run)
MLearners = c('S-BART','T-BART','X-BART',
              'R-LASSO', 'R-BOOST', 'CF', 'BCF',"S-RF","T-RF","X-RF")

nb_learner = length((MLearners))

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
rm(mylist)

system.time(
  
  
  for (i in 1:B) {
    
    
    gc()
    
    cat("\n\n\n\n-------- Iteration", i, "--------\n\n\n\n")
    
    
    if(i<=500){set.seed(502 + i*5)}
    if(i>500){set.seed(7502 + i*5)}

    bruit_gaussien <- rnorm(sample_size, mean = 0, sd = sqrt(hyperparams$sigma_sq))

    # Appliquer la fonction logistique
    fac = 0.5 # Même systeme de facteur multiplicatif pour le bruit que les setups 2a/2b
    # pour permettre les comparaisons
    myY <- plogis(ITE + bruit_gaussien * fac)


    
    
    ### Common mu(x) model for RLOSS evaluation
    mu_boost = xgboost::xgboost(label = myY, 
                                data = myX,           
                                max.depth = 3, 
                                nround = 300,
                                early_stopping_rounds = 4, 
                                objective = "reg:squarederror",
                                gamma = 1, verbose = F)
    
    mu_est = predict(mu_boost, myX)
    
    
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
    
    
    ###### MODELS ESTIMATION  ------------------------------------------------
    
    
    # ################ S-RF # Code does not run atm
    
    # Error : Error: 'forestry' is not an exported object 
    #                 from 'namespace:forestry'
    
    SRF <- S_RF(train_augmX, z_train, y_train)
    train_est = EstimateCate(SRF, train_augmX)
    test_est = EstimateCate(SRF, test_augmX)

    # CATT
    Results$CATT_Train_Bias[i, 'S-RF'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'S-RF'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'S-RF'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1],
                                                 z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])

    Results$CATT_Test_Bias[i, 'S-RF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'S-RF'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'S-RF'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1],
                                                z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])

    # CATC
    Results$CATC_Train_Bias[i, 'S-RF'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'S-RF'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'S-RF'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0],
                                                 z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])

    Results$CATC_Test_Bias[i, 'S-RF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'S-RF'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'S-RF'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0],
                                                z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])

    rm(SRF)
    
    
    
    
    ######################### S-BART
    #### Train
    start_time <- Sys.time()
    
    myBART <- wbart(x.train = cbind(train_augmX, z_train), y.train = y_train, 
                    x.test = cbind(test_augmX, z_test), nskip = 2000, ndpost = 4000, printevery = 6000)
    
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
    Results$CATT_Train_RLOSS[i, 'S-BART'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1], 
                                                   z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'S-BART'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'S-BART'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'S-BART'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1], 
                                                  z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'S-BART'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'S-BART'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'S-BART'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0], 
                                                   z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'S-BART'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'S-BART'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'S-BART'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0], 
                                                  z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])
    
    
    # Free up space
    rm(myBART, Y0_train, Y0_test)
    
    
    
    
    # #################### T-RF # Does not run atm
    TRF <- T_RF(train_augmX, z_train, y_train)
    train_est = EstimateCate(TRF, train_augmX)
    test_est = EstimateCate(TRF, test_augmX)

    # CATT
    Results$CATT_Train_Bias[i, 'T-RF'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'T-RF'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'T-RF'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1],
                                                 z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])

    Results$CATT_Test_Bias[i, 'T-RF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'T-RF'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'T-RF'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1],
                                                z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])

    # CATC
    Results$CATC_Train_Bias[i, 'T-RF'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'T-RF'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'T-RF'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0],
                                                 z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])

    Results$CATC_Test_Bias[i, 'T-RF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'T-RF'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'T-RF'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0],
                                                z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])

    rm(TRF)

    
    
    
    
    #######
    # Check that there is no single-valued binary variable 
    # and in case remove it for T-BART (as it would not split on it)
    check_train1 = which(apply(train_augmX[z_train == 1, ], 2, function(x) max(x) == min(x)))
    check_train0 = which(apply(train_augmX[z_train == 0, ], 2, function(x) max(x) == min(x)))
    
    check_test1 = which(apply(test_augmX[z_test == 1, ], 2, function(x) max(x) == min(x)))
    check_test0 = which(apply(test_augmX[z_test == 0, ], 2, function(x) max(x) == min(x)))
    
    check_all = c(check_train1, check_train0, check_test1, check_test0)
    
    if (length(check_all) > 0) {
      
      new_train_X = train_augmX[, -check_all]
      new_test_X = test_augmX[, -check_all]
      
    } else {
      
      new_train_X = train_augmX
      new_test_X = test_augmX
      
    }
    
    
    ######################### T-BART
    #### Train
    
    start_time <- Sys.time()
    
    myBART1_shell <- future({
      wbart(x.train = new_train_X[z_train == 1, ], y.train = y_train[z_train == 1], x.test = new_test_X[z_test == 1, ],
            nskip = 2000, ndpost = 4000, printevery = 6000)
    }, seed = T)
    
    myBART0_shell <- future({
      wbart(x.train = new_train_X[z_train == 0, ], y.train = y_train[z_train == 0], x.test = new_test_X[z_test == 0, ],
            nskip = 2000, ndpost = 4000, printevery = 6000)
    }, seed = T)  
    
    myBART1 <- value(myBART1_shell); myBART0 <- value(myBART0_shell)
    rm(myBART1_shell, myBART0_shell)
    
    # (predict counterfactual)
    Y1_0_shell <- future({
      predict(myBART1, newdata = new_train_X[z_train == 0, ])
    }, seed = T)
    
    Y0_1 <- predict(myBART0, newdata = new_train_X[z_train == 1, ])
    
    Y1_0 <- value(Y1_0_shell)
    rm(Y1_0_shell)
    
    
    Y1_train <- y_train
    Y1_train[z_train == 1] <- myBART1$yhat.train.mean
    Y1_train[z_train == 0] <- colMeans(Y1_0)
    
    Y0_train <- y_train
    Y0_train[z_train == 0] <- myBART0$yhat.train.mean
    Y0_train[z_train == 1] <- colMeans(Y0_1)
    
    # Measure time needed to execute X code (use value from T code)
    end_time <- Sys.time()
    execution_time_T_shared_with_X <- end_time - start_time
    
    # Store ITE estimates
    train_est = Y1_train - Y0_train
    
    
    ##### Test
    # (predict counterfactual)
    Y1_0_shell <- future({
      predict(myBART1, newdata = new_test_X[z_test == 0, ])
    })
    
    Y0_1 <- predict(myBART0, newdata = new_test_X[z_test == 1, ])
    
    Y1_0 <- value(Y1_0_shell)
    rm(Y1_0_shell)
    
    Y1_test <- y_test
    Y1_test[z_test == 1] <- myBART1$yhat.test.mean
    Y1_test[z_test == 0] <- colMeans(Y1_0)
    
    Y0_test <- y_test
    Y0_test[z_test == 0] <- myBART0$yhat.test.mean
    Y0_test[z_test == 1] <- colMeans(Y0_1)
    
    # Store ITE estimates
    test_est = Y1_test - Y0_test 
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'T-BART'] = execution_time
    
    
    # CATT
    Results$CATT_Train_Bias[i, 'T-BART'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'T-BART'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'T-BART'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1], 
                                                   z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'T-BART'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'T-BART'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'T-BART'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1], 
                                                  z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'T-BART'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'T-BART'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'T-BART'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0], 
                                                   z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'T-BART'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'T-BART'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'T-BART'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0], 
                                                  z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])
    
    # Remove garbage
    rm(myBART1, myBART0, Y0_1, Y1_0)
    
    
    
    
    # #################### X-RF
    # Remove propensity score
    XRF <- X_RF(train_augmX[,-ncol(train_augmX)], z_train, y_train)
    train_est = EstimateCate(XRF, train_augmX[,-ncol(train_augmX)])
    test_est = EstimateCate(XRF, test_augmX[,-ncol(test_augmX)])

    # CATT
    Results$CATT_Train_Bias[i, 'X-RF'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'X-RF'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'X-RF'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1],
                                                 z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])

    Results$CATT_Test_Bias[i, 'X-RF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'X-RF'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'X-RF'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1],
                                                z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])

    # CATC
    Results$CATC_Train_Bias[i, 'X-RF'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'X-RF'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'X-RF'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0],
                                                 z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])

    Results$CATC_Test_Bias[i, 'X-RF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'X-RF'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'X-RF'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0],
                                                z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])

    rm(XRF)
    
    
    
    ################# X-BART
    # Use T-BART estimates
    start_time <- Sys.time()
    
    D_1 = y_train[z_train == 1] - Y0_train[z_train == 1]
    D_0 = Y1_train[z_train == 0] - y_train[z_train == 0]
    
    
    myBART1_shell <- future({
      wbart(x.train = new_train_X[z_train == 1, -ncol(new_train_X)], 
            y.train = D_1, 
            nskip = 2000, ndpost = 4000, printevery = 6000)
    }, seed = T)
    
    myBART0_shell <- future({
      wbart(x.train = new_train_X[z_train == 0, -ncol(new_train_X)], 
            y.train = D_0, 
            nskip = 2000, ndpost = 4000, printevery = 6000)
    }, seed = T)  
    
    
    myBART1 <- value(myBART1_shell); myBART0 <- value(myBART0_shell)
    rm(myBART1_shell, myBART0_shell)
    
    
    # Train
    tau_1_shell <- future({
      predict(myBART1, 
              newdata = new_train_X[, -ncol(new_train_X)])
    }, seed = T)
    
    
    tau_0_train <- predict(myBART0, newdata = new_train_X[, -ncol(new_train_X)])
    tau_1_train <- value(tau_1_shell)
    
    rm(tau_1_shell)
    
    
    # Test
    tau_1_shell <- future({
      predict(myBART1, 
              newdata = new_test_X[, -ncol(new_test_X)])
    }, seed = T)
    
    
    tau_0_test <- predict(myBART0, newdata = new_test_X[, -ncol(new_test_X)])
    tau_1_test <- value(tau_1_shell)
    
    rm(tau_1_shell)
    
    
    # Final CATE estimates
    train_est = PS_est[smp_split == 1]*colMeans(tau_0_train) + (1 - PS_est[smp_split == 1])*colMeans(tau_1_train)
    test_est = PS_est[smp_split == 2]*colMeans(tau_0_test) + (1 - PS_est[smp_split == 2])*colMeans(tau_1_test)
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'X-BART'] = execution_time + execution_time_T_shared_with_X
    
    
    # CATT
    Results$CATT_Train_Bias[i, 'X-BART'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'X-BART'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'X-BART'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1], 
                                                   z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'X-BART'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'X-BART'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'X-BART'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1], 
                                                  z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'X-BART'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'X-BART'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'X-BART'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0], 
                                                   z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'X-BART'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'X-BART'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'X-BART'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0], 
                                                  z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])
    
    rm(myBART1, myBART0, tau_0_test, tau_0_train, tau_1_test, tau_1_train)
    
    
    
    
    ######################## R-Lasso Regression
    # No estimated PS as 
    start_time <- Sys.time()
    
    RLASSO <- rlasso(x = train_augmX[, -ncol(train_augmX)], w = z_train, y = y_train,
                     lambda_choice = "lambda.min", rs = FALSE)
    
    train_est = predict(RLASSO, train_augmX[, -ncol(train_augmX)])
    test_est = predict(RLASSO, test_augmX[, -ncol(train_augmX)])
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'R-LASSO'] = execution_time
    
    # CATT
    Results$CATT_Train_Bias[i, 'R-LASSO'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'R-LASSO'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'R-LASSO'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1], 
                                                    z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'R-LASSO'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'R-LASSO'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'R-LASSO'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1], 
                                                   z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'R-LASSO'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'R-LASSO'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'R-LASSO'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0], 
                                                    z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'R-LASSO'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'R-LASSO'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'R-LASSO'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0], 
                                                   z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])  
    
    
    rm(RLASSO)    
    
    
    
    
    ######################## R-BOOST Regression
    # No estimated PS as 
    start_time <- Sys.time()
    
    RBOOST <- rboost(x = train_augmX[, -ncol(train_augmX)], w = z_train, y = y_train, nthread=1)
    
    train_est = predict(RBOOST, train_augmX[, -ncol(train_augmX)])
    test_est = predict(RBOOST, test_augmX[, -ncol(train_augmX)])
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'R-BOOST'] = execution_time
    
    # CATT
    Results$CATT_Train_Bias[i, 'R-BOOST'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'R-BOOST'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'R-BOOST'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1], 
                                                    z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'R-BOOST'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'R-BOOST'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'R-BOOST'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1], 
                                                   z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'R-BOOST'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'R-BOOST'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'R-BOOST'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0], 
                                                    z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'R-BOOST'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'R-BOOST'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'R-BOOST'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0], 
                                                   z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])
    
    rm(RBOOST)
    
    
    
    
    ######################### Causal RF (Athey, Wagner 2019)
    # Train
    start_time <- Sys.time()
    
    CRF <- causal_forest(train_augmX[, -ncol(train_augmX)], y_train, z_train, tune.parameters = "all")
    
    train_est = predict(CRF)[, "predictions"]
    test_est = predict(CRF, newdata =  test_augmX[, -ncol(test_augmX)])[, "predictions"]
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'CF'] = execution_time
    
    
    # CATT
    Results$CATT_Train_Bias[i, 'CF'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'CF'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'CF'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1], 
                                               z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'CF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'CF'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'CF'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1], 
                                              z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'CF'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'CF'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'CF'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0], 
                                               z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'CF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'CF'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'CF'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0], 
                                              z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])
    
    # Remove garbage
    rm(CRF)
    
    
    
    
    
    ######################### BCF 
    #### Train
    
    start_time <- Sys.time()
    
    mybcf <- 
      SparseBCF(y = y_train, 
                z = z_train, 
                x_control = train_augmX[, -ncol(train_augmX)], 
                pihat = PS_est[smp_split == 1], 
                OOB = T, 
                sparse = F,
                x_pred_mu = test_augmX[, -ncol(train_augmX)], 
                pi_pred = PS_est[smp_split == 2], 
                x_pred_tau = test_augmX[, -ncol(train_augmX)], 
                nsim = 6000, nburn = 4000)
    
    
    train_est = colMeans(mybcf$tau)
    test_est = colMeans(mybcf$tau_pred)
    
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    Liste_time$execution_time[i, 'BCF'] = execution_time
    
    
    # CATT
    Results$CATT_Train_Bias[i, 'BCF'] = bias(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_PEHE[i, 'BCF'] = PEHE(Train_CATT, train_est[z_train == 1])
    Results$CATT_Train_RLOSS[i, 'BCF'] = r_loss(y_train[z_train == 1], mu_est[smp_split == 1][z_train == 1], 
                                                z_train[z_train == 1], PS_est[smp_split == 1][z_train == 1], train_est[z_train == 1])
    
    Results$CATT_Test_Bias[i, 'BCF'] = bias(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_PEHE[i, 'BCF'] = PEHE(Test_CATT, test_est[z_test == 1])
    Results$CATT_Test_RLOSS[i, 'BCF'] = r_loss(y_test[z_test == 1], mu_est[smp_split == 2][z_test == 1], 
                                               z_test[z_test == 1], PS_est[smp_split == 2][z_test == 1], test_est[z_test == 1])
    
    # CATC
    Results$CATC_Train_Bias[i, 'BCF'] = bias(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_PEHE[i, 'BCF'] = PEHE(Train_CATC, train_est[z_train == 0])
    Results$CATC_Train_RLOSS[i, 'BCF'] = r_loss(y_train[z_train == 0], mu_est[smp_split == 1][z_train == 0], 
                                                z_train[z_train == 0], PS_est[smp_split == 1][z_train == 0], train_est[z_train == 0])
    
    Results$CATC_Test_Bias[i, 'BCF'] = bias(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_PEHE[i, 'BCF'] = PEHE(Test_CATC, test_est[z_test == 0])
    Results$CATC_Test_RLOSS[i, 'BCF'] = r_loss(y_test[z_test == 0], mu_est[smp_split == 2][z_test == 0], 
                                               z_test[z_test == 0], PS_est[smp_split == 2][z_test == 0], test_est[z_test == 0])
    
    
    # Remove garbage
    rm(mybcf)
    
    
  }
  
)

sapply( names(Results), function(x) colMeans(Results[[x]]) )
sapply( names(Results), function(x) apply(Results[[x]], 2, function(y) MC_se(y, B)) )

print(Liste_time)
sapply( names(Liste_time), function(x) colMeans(Liste_time[[x]]) )


# Save Results --------------------------------------------------

directory_path <- "./Results"

if (!dir.exists(directory_path)) {
  dir.create(directory_path, recursive = TRUE)
}

invisible(
  sapply(names(Results), 
         function(x) write.csv(Results[[x]], 
                               file=paste0(getwd(), "/Results/Logit_", B, "_", x, "_fac_",fac, "_Nsize_",sample_size,".csv") ) )
)


write.csv(sapply( names(Results), function(x) colMeans(Results[[x]]) ), 
          file = paste0(getwd(), "/Results/MeanSummary_", B, "_fac_",fac, "_Nsize_",sample_size,".csv"))

write.csv(sapply( names(Results), function(x) apply(Results[[x]], 2, function(y) MC_se(y, B)) ), 
          file = paste0(getwd(), "/Results/MCSE_Summary_", B, "_fac_",fac, "_Nsize_",sample_size, ".csv"))


