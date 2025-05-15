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
    
    
optimize_and_evaluate_S_RF <- function(train_augmX, z_train, y_train, test_augmX, z_test, y_test,
                                     Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE) {
  print("starting...")

  # Define the hyperparameter grid to search
  param_grid <- expand.grid(
    ntree = c(300,500,1000),  # Number of trees
    mtry = c(1, 2, 3, 4),  # Features to try at each split
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),  # Minimum node size for splits
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),  # Minimum node size for averages
    sample.fraction =c(0.1, 0.2, 0.3, 0.4, 0.5)  # Fraction of samples used for each tree
  )

  # Initialize variables to store best parameters and performance
  best_params <- NULL
  best_performance <- Inf
  best_model <- NULL

  # Initialize a list to store performance of all models
  all_performances <- list()

  # Iterate over each row of the parameter grid
  for (i in 1:nrow(param_grid)) {
    params <- param_grid[i, ]

    #print(params)

    #start_time <- Sys.time()
    SRF <- S_RF(train_augmX, z_train, y_train,
               mu.forestry = list(
                 relevant.Variable = 1:ncol(train_augmX),
                 ntree = params$ntree,
                 replace = TRUE,
                 sample.fraction = params$sample.fraction,
                 mtry = params$mtry,
                 nodesizeSpl = params$nodesizeSpl,
                 nodesizeAvg = params$nodesizeAvg,
                 splitratio = 0.5,
                 middleSplit = FALSE
               ),
               nthread = 0)

    # train_est = EstimateCate(SRF, train_augmX)
    test_est = EstimateCate(SRF, test_augmX)

    #end_time <- Sys.time()
    #execution_time <- end_time - start_time


    #print(paste0("S-RF_execution_time : ", execution_time))

    # CATT
    CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])
    #print("CATT_Test_PEHE")
    #print(CATT_Test_PEHE)

    # Store the performance of the current model
    all_performances[[i]] <- list(params = params, performance = CATT_Test_PEHE)

    if (CATT_Test_PEHE < best_performance) {
      best_performance = CATT_Test_PEHE
      best_params = params
      best_model = SRF
      if (verbose){
      print("params")
      print(params)
      print("CATT_Test_PEHE")
      print(CATT_Test_PEHE)
      }
    }

    rm(SRF)
  }

  return(list(best_model = best_model, best_params = best_params, best_performance = best_performance, all_performances = all_performances))
}

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

optimize_and_evaluate_S_RF_2 <- function(train_augmX, z_train, y_train, test_augmX, z_test, y_test,
                                       Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE) {
  sink()
  print("starting sequential optimization...")

  # Initialize best parameters with default values
  best_params <- list(
    ntree = 1000,
    mtry = 2,
    nodesizeSpl = 1,
    nodesizeAvg = 1,
    sample.fraction = 0.2
  )

  best_performance <- Inf
  best_model <- NULL

  # Define parameter ranges to test sequentially
  param_ranges <- list(
    mtry = c(1, 2, 3, 4),
    sample.fraction = c(0.01, 0.05,0.1, 0.2, 0.3, 0.4, 0.5),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  )

  # We'll keep ntree fixed as in your original code
  #best_params$ntree <- 1000

  # Sequential optimization
  for (param_name in names(param_ranges)) {
    print(paste("Optimizing", param_name))
    current_values <- param_ranges[[param_name]]

    for (value in current_values) {
      # Update the parameter to test
      params <- best_params
      params[[param_name]] <- value

      #start_time <- Sys.time()
      SRF <- S_RF(train_augmX, z_train, y_train,
                 mu.forestry = list(
                   relevant.Variable = 1:ncol(train_augmX),
                   ntree = params$ntree,
                   replace = TRUE,
                   sample.fraction = params$sample.fraction,
                   mtry = params$mtry,
                   nodesizeSpl = params$nodesizeSpl,
                   nodesizeAvg = params$nodesizeAvg,
                   splitratio = 0.5,
                   middleSplit = FALSE
                 ),
                 nthread = 0)

      test_est = EstimateCate(SRF, test_augmX)
      #end_time <- Sys.time()
      #execution_time <- end_time - start_time

      # print(paste0("S-RF_execution_time : ", execution_time))

      # CATT
      CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])

      if (0) {
        print("Current params")
        print(params)
        print("CATT_Test_PEHE")
        print(CATT_Test_PEHE)
      }

      # Update best parameters if current performance is better
      if (CATT_Test_PEHE < best_performance) {
        best_performance = CATT_Test_PEHE
        best_params <- params
        best_model = SRF
        if (verbose){
          print("best_performance")
          print(best_performance)
        }
      }

      rm(SRF)
    }
  }

  return(list(best_model = best_model, best_params = best_params, best_performance = best_performance))
}


if (1){
for (i in 1:1){
cat("\nS_RF2 bla\n")
cat(i)
start_time_RF2 <- Sys.time()
result <- optimize_and_evaluate_S_RF_2(
  train_augmX, z_train, y_train, test_augmX, z_test, y_test,
  Train_CATT, verbose=FALSE
)

end_time_RF2 <- Sys.time()
execution_time_RF2 <- end_time_RF2 - start_time_RF2
print(paste0("S-RF_execution_time : ", execution_time_RF2))
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

optimize_and_evaluate_S_RF_3 <- function(train_augmX, z_train, y_train, test_augmX, z_test, y_test,
                                       Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE,
                                       performance_threshold = 0.9) {
  print("starting less aggressive sequential optimization...")

  current_params <- list(
    ntree = 1000,
    mtry = 2,
    nodesizeSpl = 1,
    nodesizeAvg = 1,
    sample.fraction = 0.2
  )

  best_performance <- Inf
  best_model <- NULL

  param_ranges <- list(
    sample.fraction = c(0.1, 0.2, 0.3, 0.4, 0.5),
    mtry = c(1, 2, 3, 4),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(300, 500, 1000, 1500, 2000, 3000, 5000, 10000)
  )

  acceptable_params <- list()
  acceptable_performances <- list()

  # Track how many unique parameter sets we've tried
  tried_combinations <- list()

  for (param_name in names(param_ranges)) {
    print(paste("Optimizing", param_name))
    current_values <- param_ranges[[param_name]]

    param_performances <- list()

    for (value in current_values) {
      test_params <- current_params
      test_params[[param_name]] <- value

      if (length(acceptable_params) > 0) {
        for (i in 1:length(acceptable_params)) {
          params <- acceptable_params[[i]]
          params[[param_name]] <- value

          # Hash the combination to avoid duplication
          key <- paste(unlist(params), collapse = "_")
          if (key %in% names(tried_combinations)) next
          tried_combinations[[key]] <- TRUE

          #start_time <- Sys.time()
          SRF <- S_RF(train_augmX, z_train, y_train,
                     mu.forestry = list(
                       relevant.Variable = 1:ncol(train_augmX),
                       ntree = params$ntree,
                       replace = TRUE,
                       sample.fraction = params$sample.fraction,
                       mtry = params$mtry,
                       nodesizeSpl = params$nodesizeSpl,
                       nodesizeAvg = params$nodesizeAvg,
                       splitratio = 0.5,
                       middleSplit = FALSE
                     ),
                     nthread = 0)

          test_est = EstimateCate(SRF, test_augmX)
          #end_time <- Sys.time()

          CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])

          param_performances[[length(param_performances) + 1]] <- list(
            params = params,
            performance = CATT_Test_PEHE
          )

          if (CATT_Test_PEHE < best_performance) {
            best_performance = CATT_Test_PEHE
            best_model = SRF
          }

          rm(SRF)
        }
      } else {
        params <- test_params

        key <- paste(unlist(params), collapse = "_")
        if (key %in% names(tried_combinations)) next
        tried_combinations[[key]] <- TRUE

        #start_time <- Sys.time()
        SRF <- S_RF(train_augmX, z_train, y_train,
                   mu.forestry = list(
                     relevant.Variable = 1:ncol(train_augmX),
                     ntree = params$ntree,
                     replace = TRUE,
                     sample.fraction = params$sample.fraction,
                     mtry = params$mtry,
                     nodesizeSpl = params$nodesizeSpl,
                     nodesizeAvg = params$nodesizeAvg,
                     splitratio = 0.5,
                     middleSplit = FALSE
                   ),
                   nthread = 0)

        test_est = EstimateCate(SRF, test_augmX)
        #end_time <- Sys.time()

        CATT_Test_PEHE = PEHE(Test_CATT, test_est[z_test == 1])

        param_performances[[length(param_performances) + 1]] <- list(
          params = params,
          performance = CATT_Test_PEHE
        )

        if (CATT_Test_PEHE < best_performance) {
          best_performance = CATT_Test_PEHE
          best_model = SRF
        }

        rm(SRF)
      }
    }

    current_best <- min(sapply(param_performances, function(x) x$performance))

    acceptable_params <- list()
    acceptable_performances <- list()

    for (i in 1:length(param_performances)) {
      perf_ratio <- param_performances[[i]]$performance / current_best
      if (perf_ratio <= 1 / performance_threshold) {
        acceptable_params[[length(acceptable_params) + 1]] <- param_performances[[i]]$params
        acceptable_performances[[length(acceptable_performances) + 1]] <- param_performances[[i]]$performance
      }
    }

    if (verbose) {
      print(paste("Keeping", length(acceptable_params), "parameter combinations for", param_name))
      print(paste("Best performance:", current_best))
    }

    if (length(acceptable_params) > 0) {
      best_single_param <- acceptable_params[[which.min(sapply(acceptable_performances, function(x) x))]]
      current_params[[param_name]] <- best_single_param[[param_name]]
    }
  }

  if (length(acceptable_params) > 0) {
    best_index <- which.min(sapply(acceptable_performances, function(x) x))
    best_params <- acceptable_params[[best_index]]
    best_performance <- acceptable_performances[[best_index]]
  } else {
    best_params <- current_params
  }

  # Compute total possible combinations (full grid)
  total_possible <- prod(sapply(param_ranges, length))
  total_tried <- length(tried_combinations)

  cat("Tried", total_tried, "unique combinations out of", total_possible, "possible.\n")

  return(list(best_model = best_model, best_params = best_params, best_performance = best_performance,
              acceptable_params = acceptable_params, acceptable_performances = acceptable_performances))
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

optimize_and_evaluate_T_RF <- function(train_augmX, z_train, y_train,
                                       test_augmX, z_test, y_test,
                                       Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE) {
  sink()
  print("Starting sequential optimization for T-RF...")

  # Initialize best parameters
  best_mu0 <- list(
    ntree = 1000,
    mtry = ncol(train_augmX),
    nodesizeSpl = 1,
    nodesizeAvg = 3,
    sample.fraction = 0.9
  )

  best_mu1 <- best_mu0  # same defaults, separate optimization
  best_performance <- Inf
  best_model <- NULL

  # Define parameter grid
  param_ranges <- list(
    mtry = c(1, 2, 3, 4, ncol(train_augmX)),
    sample.fraction = c(0.05, 0.1, 0.2, 0.5, 0.9),
    nodesizeSpl = c(1, 3, 5, 10, 15),
    nodesizeAvg = c(1, 3, 5, 10, 15),
    ntree = c(300, 500, 1000, 1500)
  )

  # ---- Optimize mu0 ----
  print("---- Optimize mu0 ----")
  for (param_name in names(param_ranges)) {
    print(paste("Optimizing mu0:", param_name))
    for (value in param_ranges[[param_name]]) {
      mu0_params <- best_mu0
      mu0_params[[param_name]] <- value

      TRF <- T_RF(
        feat = train_augmX,
        tr = z_train,
        yobs = y_train,
        nthread = nthread,
        verbose = FALSE,
        mu0.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = mu0_params$ntree,
          replace = TRUE,
          sample.fraction = mu0_params$sample.fraction,
          mtry = mu0_params$mtry,
          nodesizeSpl = mu0_params$nodesizeSpl,
          nodesizeAvg = mu0_params$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        ),
        mu1.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_mu1$ntree,
          replace = TRUE,
          sample.fraction = best_mu1$sample.fraction,
          mtry = best_mu1$mtry,
          nodesizeSpl = best_mu1$nodesizeSpl,
          nodesizeAvg = best_mu1$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        )
      )

      test_est <- EstimateCate(TRF, test_augmX)
      PEHE_val <- PEHE(Test_CATT, test_est[z_test == 1])

      if (PEHE_val < best_performance) {
        best_performance <- PEHE_val
        best_mu0 <- mu0_params
        best_model <- TRF
        if (verbose) {
          print("Updated best performance (mu0):")
          print(best_performance)
        }
      }

      rm(TRF)
    }
  }

  # ---- Optimize mu1 ----
  print("---- Optimize mu1 ----")
  for (param_name in names(param_ranges)) {
    print(paste("Optimizing mu1:", param_name))
    for (value in param_ranges[[param_name]]) {
      mu1_params <- best_mu1
      mu1_params[[param_name]] <- value

      TRF <- T_RF(
        feat = train_augmX,
        tr = z_train,
        yobs = y_train,
        nthread = nthread,
        verbose = FALSE,
        mu0.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_mu0$ntree,
          replace = TRUE,
          sample.fraction = best_mu0$sample.fraction,
          mtry = best_mu0$mtry,
          nodesizeSpl = best_mu0$nodesizeSpl,
          nodesizeAvg = best_mu0$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        ),
        mu1.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = mu1_params$ntree,
          replace = TRUE,
          sample.fraction = mu1_params$sample.fraction,
          mtry = mu1_params$mtry,
          nodesizeSpl = mu1_params$nodesizeSpl,
          nodesizeAvg = mu1_params$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        )
      )

      test_est <- EstimateCate(TRF, test_augmX)
      PEHE_val <- PEHE(Test_CATT, test_est[z_test == 1])

      if (PEHE_val < best_performance) {
        best_performance <- PEHE_val
        best_mu1 <- mu1_params
        best_model <- TRF
        if (verbose) {
          print("Updated best performance (mu1):")
          print(best_performance)
        }
      }

      rm(TRF)
    }
  }

  return(list(
    best_model = best_model,
    best_mu0_params = best_mu0,
    best_mu1_params = best_mu1,
    best_performance = best_performance
  ))
}


cat("\n\n\n\n")
print("T_RF")
start_time_RF <- Sys.time()
result <- optimize_and_evaluate_T_RF(
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