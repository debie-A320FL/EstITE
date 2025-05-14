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


data <- read.csv("./../Setup 1a/Data/simulated_100M_data.csv")

set.seed(123)

data = data[sample(nrow(data)),]
size_sample = 1e7
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


for (i in 1:3){
start_time <- Sys.time()
    
RLASSO <- rlasso(x = train_augmX[, -ncol(train_augmX)], w = z_train, y = y_train,
                    lambda_choice = "lambda.min", rs = FALSE)

train_est = predict(RLASSO, train_augmX[, -ncol(train_augmX)])
test_est = predict(RLASSO, test_augmX[, -ncol(train_augmX)])

end_time <- Sys.time()
execution_time <- end_time - start_time
print("execution_time")
print(execution_time)

perf = PEHE(Test_CATT, test_est[z_test == 1])
print("perf")
print(perf)
}