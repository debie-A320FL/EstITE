library(caret)
library(parallel)
library(doParallel)

# Function to optimize S_RF hyperparameters
optimize_S_RF <- function(feat, tr, yobs, nfolds = 5, nthread = 0) {
  # Define the hyperparameter grid to search
  param_grid <- expand.grid(
    ntree = c(500, 1000, 1500, 2000),  # Number of trees
    mtry = c(floor(ncol(feat)/3), floor(ncol(feat)/2), floor(sqrt(ncol(feat))), ncol(feat)),  # Features to try at each split
    nodesizeSpl = c(1, 3, 5),  # Minimum node size for splits
    nodesizeAvg = c(3, 5, 10),  # Minimum node size for averages
    sample.fraction = c(0.6, 0.7, 0.8, 0.9),  # Fraction of samples used for each tree
    splitratio = c(0.3, 0.5, 0.7)  # Split ratio for categorical variables
  )

  # Initialize variables to store best parameters and performance
  best_params <- NULL
  best_performance <- Inf
  best_model <- NULL

  # Set up parallel processing
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)

  # Create folds for cross-validation
  folds <- createFolds(yobs, k = nfolds)

  # Perform grid search
  for (i in 1:nrow(param_grid)) {
    params <- param_grid[i, ]

    # Cross-validation performance
    cv_performance <- foreach(j = 1:nfolds, .combine = c) %dopar% {
      # Get training and validation indices
      train_idx <- folds[[j]]
      val_idx <- setdiff(1:length(yobs), train_idx)

      # Train the model
      model <- S_RF(
        feat[train_idx, ],
        tr[train_idx],
        yobs[train_idx],
        nthread = nthread,
        verbose = FALSE,
        mu.forestry = list(
          relevant.Variable = 1:ncol(feat),
          ntree = params$ntree,
          replace = TRUE,
          sample.fraction = params$sample.fraction,
          mtry = params$mtry,
          nodesizeSpl = params$nodesizeSpl,
          nodesizeAvg = params$nodesizeAvg,
          splitratio = params$splitratio,
          middleSplit = FALSE
        )
      )

      # Predict on validation set
      pred <- predict(model, feat[val_idx, ])

      # Calculate performance metric (MSE in this case)
      mse <- mean((pred - yobs[val_idx])^2)
    }

    # Calculate average performance across folds
    avg_performance <- mean(cv_performance)

    # Update best parameters if current configuration is better
    if (avg_performance < best_performance) {
      best_performance <- avg_performance
      best_params <- params

      # Retrain the model with best parameters on full data
      best_model <- S_RF(
        feat,
        tr,
        yobs,
        nthread = nthread,
        verbose = FALSE,
        mu.forestry = list(
          relevant.Variable = 1:ncol(feat),
          ntree = params$ntree,
          replace = TRUE,
          sample.fraction = params$sample.fraction,
          mtry = params$mtry,
          nodesizeSpl = params$nodesizeSpl,
          nodesizeAvg = params$nodesizeAvg,
          splitratio = params$splitratio,
          middleSplit = FALSE
        )
      )
    }

    cat("Tested configuration", i, "of", nrow(param_grid),
        "- Performance:", avg_performance, "\n")
  }

  # Stop parallel processing
  stopCluster(cl)

  # Return the best model and parameters
  list(
    best_model = best_model,
    best_params = best_params,
    best_performance = best_performance
  )
}

# Example usage:
# Assuming you have your features (feat), treatment (tr), and outcome (yobs) ready
# result <- optimize_S_RF(feat, tr, yobs)
# best_model <- result$best_model
# best_params <- result$best_params
