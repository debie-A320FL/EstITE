curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./../..')

generate_summary_tables <- function(Nsize_vector = c(500,1000,4000), num_ACTG = 80, num_GP = 80) {
  
  # Initialize empty lists to store results
  mean_train_list <- list()
  sd_train_list <- list()
  mean_test_list <- list()
  sd_test_list <- list()
  
  # Loop through each N value
  for (N in Nsize_vector) {
    # Construct the file paths dynamically
    file_path_train_actg <- paste0("./Results/Logit_", num_ACTG, "_CATT_Train_PEHE_Nsize_", N, ".csv")
    file_path_train_gp <- paste0("./Results/GP_", num_GP, "_CATT_Train_PEHE_Nsize_", N, ".csv")
    file_path_test_actg <- paste0("./Results/Logit_", num_ACTG, "_CATT_Test_PEHE_Nsize_", N, ".csv")
    file_path_test_gp <- paste0("./Results/GP_", num_GP, "_CATT_Test_PEHE_Nsize_", N, ".csv")
    
    # Read the CSV files using the constructed file paths
    PEHE_Train <- cbind(read.csv(file_path_train_actg), read.csv(file_path_train_gp))
    PEHE_Test <- cbind(read.csv(file_path_test_actg), read.csv(file_path_test_gp))
    
    PEHE_Train$X = NULL; PEHE_Test$X = NULL
    
    PEHE_Train <- reshape(data = PEHE_Train, varying = list(names(PEHE_Train)), timevar = "Model",
                          times = names(PEHE_Train), direction = "long", v.names = "PEHE")
    PEHE_Test <- reshape(data = PEHE_Test, varying = list(names(PEHE_Test)), timevar = "Model",
                         times = names(PEHE_Test), direction = "long", v.names = "PEHE")
    
    m_order <- c("BCF", "NSGP", "CMGP", "CF", "R-BOOST", "R-LASSO",
                 "X-BART", "T-BART", "S-BART", 'S-RF', "T-RF", "X-RF", "Logistic")
    
    PEHE_Train <-
      PEHE_Train %>%
      select(-id) %>%
      mutate(Model = gsub("[.]", "-", Model),
             Model = factor(Model,
                            levels = m_order,
                            ordered = TRUE))
    
    PEHE_Test <-
      PEHE_Test %>%
      select(-id) %>%
      mutate(Model = gsub("[.]", "-", Model),
             Model = factor(Model,
                            levels = m_order,
                            ordered = TRUE))
    
    # Calculate mean and sd for each model and fac value
    mean_train <- PEHE_Train %>%
      group_by(Model) %>%
      summarise(mean_PEHE = format(mean(PEHE, na.rm = TRUE), scientific = TRUE, digits = 2)) %>%
      pull(mean_PEHE)
    
    sd_train <- PEHE_Train %>%
      group_by(Model) %>%
      summarise(sd_PEHE = format(sd(PEHE, na.rm = TRUE), scientific = TRUE, digits = 2)) %>%
      pull(sd_PEHE)
    
    mean_test <- PEHE_Test %>%
      group_by(Model) %>%
      summarise(mean_PEHE = format(mean(PEHE, na.rm = TRUE), scientific = TRUE, digits = 2)) %>%
      pull(mean_PEHE)
    
    sd_test <- PEHE_Test %>%
      group_by(Model) %>%
      summarise(sd_PEHE = format(sd(PEHE, na.rm = TRUE), scientific = TRUE, digits = 2)) %>%
      pull(sd_PEHE)
    
    # Store the results in the lists
    mean_train_list[[as.character(N)]] <- mean_train
    sd_train_list[[as.character(N)]] <- sd_train
    mean_test_list[[as.character(N)]] <- mean_test
    sd_test_list[[as.character(N)]] <- sd_test
  }
  
  # Convert lists to data frames
  mean_train_df <- do.call(cbind, mean_train_list)
  sd_train_df <- do.call(cbind, sd_train_list)
  mean_test_df <- do.call(cbind, mean_test_list)
  sd_test_df <- do.call(cbind, sd_test_list)
  
  # Add row names (model names)
  rownames(mean_train_df) <- m_order
  rownames(sd_train_df) <- m_order
  rownames(mean_test_df) <- m_order
  rownames(sd_test_df) <- m_order
  
  # Print the tables
  cat("\nMean PEHE for Training Data:\n\n\n")
  print(mean_train_df)
  
  cat("\nStandard Deviation PEHE for Training Data:\n\n\n")
  print(sd_train_df)
  
  cat("\nMean PEHE for Test Data:\n\n\n")
  print(mean_test_df)
  
  cat("\nStandard Deviation PEHE for Test Data:\n\n\n")
  print(sd_test_df)
  
  # Optionally, return the data frames
  list(mean_train = mean_train_df, sd_train = sd_train_df, mean_test = mean_test_df, sd_test = sd_test_df)
}

# Example usage
results <- generate_summary_tables(Nsize_vector = c(500,1000,4000))
