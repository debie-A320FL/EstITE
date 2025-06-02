source("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5/Code/R/Sample_creation.R")

optimize_and_evaluate_S_RF <- function(train_augmX, z_train, y_train, test_augmX, z_test, y_test,
                                     Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE,param_grid= NULL){
  print("starting...")

  # Define the hyperparameter grid to search
  if (is.null(param_grid)){
  param_grid <- expand.grid(
    ntree = c(300,500,1000),  # Number of trees
    mtry = c(1, 2, 3, 4),  # Features to try at each split
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),  # Minimum node size for splits
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),  # Minimum node size for averages
    sample.fraction =c(0.1, 0.2, 0.3, 0.4, 0.5)  # Fraction of samples used for each tree
  )
  }

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

optimize_and_evaluate_S_RF_2 <- function(train_augmX, z_train, y_train, test_augmX, z_test, y_test,
                                       Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE) {

  #print("starting sequential optimization...")

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
    sample.fraction = c(0.01, 0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  )

  # We'll keep ntree fixed as in your original code
  #best_params$ntree <- 1000

  # Sequential optimization
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing", param_name))
    }
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

optimize_and_evaluate_T_RF <- function(train_augmX, z_train, y_train,
                                       test_augmX, z_test, y_test,
                                       Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE) {
  #print("Starting sequential optimization for T-RF...")

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
    mtry = c(1, 2, 3, 4),
    sample.fraction = c(0.01, 0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  )

  # ---- Optimize mu0 ----
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing mu0:", param_name))
    }
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
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing mu1:", param_name))
    }
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

optimize_and_evaluate_X_RF <- function(train_augmX, z_train, y_train,
                                       test_augmX, z_test, y_test,
                                       Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE) {
  #print("Starting sequential optimization for X-RF...")

  # Set default values
  default_params <- list(
    ntree = 1000,
    mtry = round(ncol(train_augmX) * 0.5),
    nodesizeSpl = 2,
    nodesizeAvg = 1,
    sample.fraction = 0.5
  )

  best_mu <- default_params
  best_tau <- default_params
  best_e <- list(
    ntree = 500,
    mtry = ncol(train_augmX),
    nodesizeSpl = 11,
    nodesizeAvg = 33,
    sample.fraction = 0.5
  )

  best_performance <- Inf
  best_model <- NULL

  param_ranges <- list(
    mtry = c(1, 2, 3, 4),
    sample.fraction = c(0.01, 0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  )

  ### -------- Optimize mu.forestry --------
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing mu.forestry:", param_name))
    }
    for (value in param_ranges[[param_name]]) {
      mu_params <- best_mu
      mu_params[[param_name]] <- value

      XRF <- X_RF(
        feat = train_augmX,
        tr = z_train,
        yobs = y_train,
        predmode = "propmean",
        nthread = nthread,
        verbose = FALSE,
        mu.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = mu_params$ntree,
          replace = TRUE,
          sample.fraction = mu_params$sample.fraction,
          mtry = mu_params$mtry,
          nodesizeSpl = mu_params$nodesizeSpl,
          nodesizeAvg = mu_params$nodesizeAvg,
          splitratio = 1,
          middleSplit = TRUE
        ),
        tau.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_tau$ntree,
          replace = TRUE,
          sample.fraction = best_tau$sample.fraction,
          mtry = best_tau$mtry,
          nodesizeSpl = best_tau$nodesizeSpl,
          nodesizeAvg = best_tau$nodesizeAvg,
          splitratio = 0.8,
          middleSplit = TRUE
        ),
        e.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_e$ntree,
          replace = TRUE,
          sample.fraction = best_e$sample.fraction,
          mtry = best_e$mtry,
          nodesizeSpl = best_e$nodesizeSpl,
          nodesizeAvg = best_e$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        )
      )

      test_est <- EstimateCate(XRF, test_augmX)
      PEHE_val <- PEHE(Test_CATT, test_est[z_test == 1])

      if (PEHE_val < best_performance) {
        best_performance <- PEHE_val
        best_mu <- mu_params
        best_model <- XRF
        if (verbose) {
          print("Updated best performance (mu):")
          print(best_performance)
        }
      }

      rm(XRF)
    }
  }

  ### -------- Optimize tau.forestry --------
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing tau.forestry:", param_name))
    }
    for (value in param_ranges[[param_name]]) {
      tau_params <- best_tau
      tau_params[[param_name]] <- value

      XRF <- X_RF(
        feat = train_augmX,
        tr = z_train,
        yobs = y_train,
        predmode = "propmean",
        nthread = nthread,
        verbose = FALSE,
        mu.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_mu$ntree,
          replace = TRUE,
          sample.fraction = best_mu$sample.fraction,
          mtry = best_mu$mtry,
          nodesizeSpl = best_mu$nodesizeSpl,
          nodesizeAvg = best_mu$nodesizeAvg,
          splitratio = 1,
          middleSplit = TRUE
        ),
        tau.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = tau_params$ntree,
          replace = TRUE,
          sample.fraction = tau_params$sample.fraction,
          mtry = tau_params$mtry,
          nodesizeSpl = tau_params$nodesizeSpl,
          nodesizeAvg = tau_params$nodesizeAvg,
          splitratio = 0.8,
          middleSplit = TRUE
        ),
        e.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_e$ntree,
          replace = TRUE,
          sample.fraction = best_e$sample.fraction,
          mtry = best_e$mtry,
          nodesizeSpl = best_e$nodesizeSpl,
          nodesizeAvg = best_e$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        )
      )

      test_est <- EstimateCate(XRF, test_augmX)

      if (any(is.na(test_est)) || length(test_est[z_test == 1]) == 0 || any(is.na(test_est[z_test == 1]))) {
        warning(paste("Skipping parameter combo due to NA in test_est or empty treated subset"))
        next
      }

      if (length(Test_CATT) != sum(z_test == 1)) {
        warning("Length mismatch between Test_CATT and test_est[z_test == 1]")
        next
      }

      PEHE_val <- tryCatch({
        PEHE(Test_CATT, test_est[z_test == 1])
      }, error = function(e) {
        warning(paste("PEHE computation failed:", e$message))
        NA
      })

      if (!is.na(PEHE_val) && PEHE_val < best_performance) {
        best_performance <- PEHE_val
        best_tau <- tau_params
        best_model <- XRF
        if (verbose) {
          print("Updated best performance (tau):")
          print(best_performance)
        }
      }

      rm(XRF)
    }
  }

  ### -------- Optimize e.forestry --------
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing e.forestry:", param_name))
    }
    for (value in param_ranges[[param_name]]) {
      e_params <- best_e
      e_params[[param_name]] <- value

      if (verbose){
        print("ntree")
        print(e_params$ntree)
      }

      XRF <- X_RF(
        feat = train_augmX,
        tr = z_train,
        yobs = y_train,
        predmode = "propmean",
        nthread = nthread,
        verbose = FALSE,
        mu.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_mu$ntree,
          replace = TRUE,
          sample.fraction = best_mu$sample.fraction,
          mtry = best_mu$mtry,
          nodesizeSpl = best_mu$nodesizeSpl,
          nodesizeAvg = best_mu$nodesizeAvg,
          splitratio = 1,
          middleSplit = TRUE
        ),
        tau.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_tau$ntree,
          replace = TRUE,
          sample.fraction = best_tau$sample.fraction,
          mtry = best_tau$mtry,
          nodesizeSpl = best_tau$nodesizeSpl,
          nodesizeAvg = best_tau$nodesizeAvg,
          splitratio = 0.8,
          middleSplit = TRUE
        ),
        e.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = e_params$ntree,
          replace = TRUE,
          sample.fraction = e_params$sample.fraction,
          mtry = e_params$mtry,
          nodesizeSpl = e_params$nodesizeSpl,
          nodesizeAvg = e_params$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        )
      )

      if (verbose){
        print("XRF succeed")
      }

      test_est <- EstimateCate(XRF, test_augmX)
      PEHE_val <- PEHE(Test_CATT, test_est[z_test == 1])

      if (PEHE_val < best_performance) {
        best_performance <- PEHE_val
        best_e <- e_params
        best_model <- XRF
        if (verbose) {
          print("Updated best performance (e):")
          print(best_performance)
        }
      }

      rm(XRF)
    }
  }

  return(list(
    best_model = best_model,
    best_mu_params = best_mu,
    best_tau_params = best_tau,
    best_e_params = best_e,
    best_performance = best_performance
  ))
}

prepare_train_data <- function(data,size_sample, hyperparams,seed = 123, train_ratio = 0.7,
                                treatment_percentile = 35, verbose = FALSE){
  set.seed(seed)

  data = data[sample(nrow(data)),]
  #size_sample = nrow(data)
  #print("size sample")
  #print(size_sample)
  data = data[1:size_sample,]

  myX <- data %>% select(-treatment, -Y) %>% as.matrix()

  if (verbose){
    cat("Searching for optimal beta_0")
  }

  beta_0 = find_optimal_beta_0(treatment_percentile/100) # transform percentile to proportion

  if (verbose){
    cat("\nFound beta_0 = ",beta_0, " to match treatment_percentile = ", treatment_percentile)
  }

  # Calculer la probabilité de traitement
  prob_treatment <- 1 / (1 + exp(-(beta_0 + hyperparams$beta_1 * myX[, "age"] + hyperparams$beta_2 * myX[, "weight"] + hyperparams$beta_3 * myX[, "comorbidities"] + hyperparams$beta_4 * myX[, "gender"])))
  myZ <- rbinom(size_sample, 1, prob_treatment)

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
  #print("estimating PS")
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
  mysplit <- c(rep(1, ceiling(train_ratio*N)), 
                rep(2, floor((1- train_ratio)*N)))
  
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

  return(list(train_augmX = train_augmX, z_train = z_train, y_train = y_train,
              test_augmX = test_augmX, z_test = z_test, y_test = y_test,
              Test_CATT = Test_CATT))
}

optimize_and_evaluate_rlasso <- function(x_train, w_train, y_train,
                                         x_test, w_test, y_test,
                                         Test_CATT,
                                         alpha_values = seq(0, 1, by = 0.1),
                                         lambda_choice = "lambda.min",
                                         verbose = TRUE) {

  best_performance <- Inf
  best_model <- NULL
  best_alpha <- NULL

  for (alpha in alpha_values) {
    if (verbose) {
      #cat("Evaluating alpha =", alpha, "\n")
    }

    # Train rlasso model
    model <- rlasso(
      x = x_train,
      w = w_train,
      y = y_train,
      alpha = alpha,
      lambda_choice = lambda_choice
    )

    # Estimate CATE on test set
    cate_test_est <- predict(model, newx = x_test)

    # Calculate PEHE (or another performance metric)
    current_pehe <- PEHE(Test_CATT, cate_test_est[w_test == 1])

    if (verbose) {
      cat("PEHE for alpha =", alpha, ":", current_pehe, "\n")
    }

    if (current_pehe < best_performance) {
      best_performance <- current_pehe
      best_model <- model
      best_alpha <- alpha
      if (verbose) {
        cat("New best alpha found:", best_alpha, "with PEHE:", best_performance, "\n")
      }
    }

    rm(model)
  }

  return(list(
    best_model = best_model,
    best_alpha = best_alpha,
    best_performance = best_performance
  ))
}

optimize_and_evaluate_X_logit_RF <- function(train_augmX, z_train, y_train,
                                       test_augmX, z_test, y_test,
                                       Test_CATT, nfolds = 5, nthread = 0, verbose=TRUE) {
  #print("Starting sequential optimization for X-RF...")

  # Set default values
  default_params <- list(
    ntree = 1000,
    mtry = 2,
    nodesizeSpl = 2,
    nodesizeAvg = 1,
    sample.fraction = 0.5
  )

  best_tau <- default_params
  best_e <- default_params

  best_performance <- Inf
  best_model <- NULL

  param_ranges <- list(
    mtry = c(1, 2, 3, 4),
    sample.fraction = c(0.01, 0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    nodesizeSpl = c(1, 3, 5, 10, 15, 20, 25, 30),
    nodesizeAvg = c(1, 3, 5, 10, 15, 20, 25, 30),
    ntree = c(1000, 1500, 2000, 3000, 5000, 10000, 300, 500)
  )


  ### -------- Optimize tau.forestry --------
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing tau.forestry:", param_name))
    }
    for (value in param_ranges[[param_name]]) {
      tau_params <- best_tau
      tau_params[[param_name]] <- value

      XRF <- X_RF(
        feat = train_augmX,
        tr = z_train,
        yobs = y_train,
        predmode = "propmean",
        nthread = nthread,
        verbose = FALSE,
        tau.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = tau_params$ntree,
          replace = TRUE,
          sample.fraction = tau_params$sample.fraction,
          mtry = tau_params$mtry,
          nodesizeSpl = tau_params$nodesizeSpl,
          nodesizeAvg = tau_params$nodesizeAvg,
          splitratio = 0.8,
          middleSplit = TRUE
        ),
        e.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_e$ntree,
          replace = TRUE,
          sample.fraction = best_e$sample.fraction,
          mtry = best_e$mtry,
          nodesizeSpl = best_e$nodesizeSpl,
          nodesizeAvg = best_e$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        )
      )

      test_est <- EstimateCate(XRF, test_augmX)

      if (any(is.na(test_est)) || length(test_est[z_test == 1]) == 0 || any(is.na(test_est[z_test == 1]))) {
        warning(paste("Skipping parameter combo due to NA in test_est or empty treated subset"))
        next
      }

      if (length(Test_CATT) != sum(z_test == 1)) {
        warning("Length mismatch between Test_CATT and test_est[z_test == 1]")
        next
      }

      PEHE_val <- tryCatch({
        PEHE(Test_CATT, test_est[z_test == 1])
      }, error = function(e) {
        warning(paste("PEHE computation failed:", e$message))
        NA
      })

      if (!is.na(PEHE_val) && PEHE_val < best_performance) {
        best_performance <- PEHE_val
        best_tau <- tau_params
        best_model <- XRF
        if (verbose) {
          print("Updated best performance (tau):")
          print(best_performance)
        }
      }

      rm(XRF)
    }
  }

  ### -------- Optimize e.forestry --------
  for (param_name in names(param_ranges)) {
    if (verbose){
    print(paste("Optimizing e.forestry:", param_name))
    }
    for (value in param_ranges[[param_name]]) {
      e_params <- best_e
      e_params[[param_name]] <- value

      if (verbose){
        print("ntree")
        print(e_params$ntree)
      }

      XRF <- X_RF(
        feat = train_augmX,
        tr = z_train,
        yobs = y_train,
        predmode = "propmean",
        nthread = nthread,
        verbose = FALSE,
        tau.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = best_tau$ntree,
          replace = TRUE,
          sample.fraction = best_tau$sample.fraction,
          mtry = best_tau$mtry,
          nodesizeSpl = best_tau$nodesizeSpl,
          nodesizeAvg = best_tau$nodesizeAvg,
          splitratio = 0.8,
          middleSplit = TRUE
        ),
        e.forestry = list(
          relevant.Variable = 1:ncol(train_augmX),
          ntree = e_params$ntree,
          replace = TRUE,
          sample.fraction = e_params$sample.fraction,
          mtry = e_params$mtry,
          nodesizeSpl = e_params$nodesizeSpl,
          nodesizeAvg = e_params$nodesizeAvg,
          splitratio = 0.5,
          middleSplit = FALSE
        )
      )

      test_est <- EstimateCate(XRF, test_augmX)
      PEHE_val <- PEHE(Test_CATT, test_est[z_test == 1])

      if (PEHE_val < best_performance) {
        best_performance <- PEHE_val
        best_e <- e_params
        best_model <- XRF
        if (verbose) {
          print("Updated best performance (e):")
          print(best_performance)
        }
      }

      rm(XRF)
    }
  }

  return(list(
    best_model = best_model,
    best_tau_params = best_tau,
    best_e_params = best_e,
    best_performance = best_performance
  ))
}