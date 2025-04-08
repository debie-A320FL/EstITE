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

MSE_100 <- function(x, y) sqrt(mean((100*x - 100*y)^2))

MC_se <- function(x, B) qt(0.975, B - 1) * sd(x) / sqrt(B)

r_loss <- function(y, mu, z, pi, tau) mean(((y - mu) - (z - pi) * tau)^2)

### OPTIONS
B = 3   # Num of Simulations

# Importer les données
# Load Data
curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./../../Data')

data <- read.csv("simulated_1M_data.csv")

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

# Ajouter une colonne pi pour la probabilité théorique
data$pi <- 1 / (1 + exp(-(mu_0 + tau * myZ)))

# Remove unused vars
rm(data, hyperparams)


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

sample_size <- round(logseq(100, nrow(data)))


vec_MSE = c()
#for (size in sample_size){
for (i in 1:length(sample_size)){
  size = sample_size[i]
  print(i)
  data_reduced = data[sample(nrow(data)),]
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
