library(ggplot2)

# Function to collect data from CSV files
collect_data_from_csv <- function(num_ACTG = 3, num_GP = 3, fac = 1, size_sample = 500) {
  file_path_time_actg <- paste0("./Results/Execution time_R_", num_ACTG, "_fac_", fac, "_Nsize_", size_sample, ".csv")
  file_path_time_gp <- paste0("./Results/Time_Nsize_", size_sample, "_B_", num_GP, ".csv")
  file_path_test_actg <- paste0("./Results/Logit_", num_ACTG, "_CATT_Test_PEHE_fac_", fac, "_Nsize_",size_sample,".csv")
  file_path_test_gp <- paste0("./Results/GP_", num_GP, "_CATT_Test_PEHE_Nsize_",size_sample,"_fac_", fac, ".csv")

  # Read the CSV files using the constructed file paths
  Time <- cbind(read.csv(file_path_time_actg), read.csv(file_path_time_gp))
  PEHE_Test <- cbind(read.csv(file_path_test_actg), read.csv(file_path_test_gp))

  L <- list(Time = Time[,-1], PEHE_Test = PEHE_Test[,-1])

  return(L)
}

# Function to create plots
create_plots <- function(sample_sizes) {
  # Initialize lists to store data
  pehe_data <- list()
  time_data <- list()

  # Collect data for each sample size
  for (size in sample_sizes) {
    res <- collect_data_from_csv(size_sample = size)
    pehe_data[[as.character(size)]] <- res$PEHE_Test
    time_data[[as.character(size)]] <- res$Time
  }

    print("pehe_data")
  print(pehe_data)

  # Calculate median for PEHE_Test
  pehe_medians <- sapply(pehe_data, function(df) {
    apply(df, 2, median, na.rm = TRUE)
  })

  # Calculate median for Time
  time_medians <- sapply(time_data, function(df) {
    apply(df, 2, median, na.rm = TRUE)
  })

    

  # Transpose the matrices to align dimensions correctly
  pehe_medians <- t(pehe_medians)
  time_medians <- t(time_medians)

  #print("pehe_medians")
  #print(pehe_medians)

  print("time_medians")
  print(time_medians)

  # Debugging: Print dimensions and column names
  #cat("Dimensions of pehe_medians:", dim(pehe_medians), "\n")
  #cat("Column names of pehe_medians:", colnames(pehe_medians), "\n")
  #cat("Dimensions of time_medians:", dim(time_medians), "\n")
  #cat("Column names of time_medians:", colnames(time_medians), "\n")

 # Convert to data frames for plotting
  pehe_df <- data.frame(
    SampleSize = rep(sample_sizes, times = ncol(pehe_medians)),
    Method = rep(colnames(pehe_medians), each = length(sample_sizes)),
    PEHE_Test = as.vector(pehe_medians)
  )

  time_df <- data.frame(
    SampleSize = rep(sample_sizes, times = ncol(time_medians)),
    Method = rep(colnames(time_medians), each = length(sample_sizes)),
    Time = as.vector(time_medians)
  )

  # Debugging: Print data frames
  #cat("pehe_df:\n")
  #print(pehe_df)

  cat("time_df:\n")
  print(time_df)

  # Create a vector of colors
  colors <- c("#1B9E77", "#070707", "#6960e6", "#e9146d", "#0de177",
            "#d4840c", "#A6761D", "#666666", "#1F77B4", "#FF7F0E",
            "#237e23", "#b34c4c", "#ad8acd")




  # Create a vector of line types
  line_types <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

  # Create PEHE_Test plot with logarithmic scale for SampleSize and PEHE_Test
  pehe_plot <- ggplot(pehe_df, aes(x = SampleSize, y = PEHE_Test, color = Method, linetype = Method)) +
    geom_point() +
    geom_line() +
    scale_x_log10(breaks = sample_sizes, labels = sample_sizes) +  # Logarithmic scale for x-axis
    scale_y_log10() +  # Logarithmic scale for y-axis
    scale_color_manual(values = colors) +  # Assign colors
    scale_linetype_manual(values = line_types) +  # Assign line types
    labs(title = "PEHE_Test vs Sample Size (Log Scale)",
         x = "Sample Size (Log Scale)",
         y = "PEHE_Test (Log Scale)",
         color = "Method") +  # Add legend title
    theme_minimal() +
    theme(legend.position = "right")  # Ensure legend is visible

  # Create Time plot with logarithmic scale for SampleSize and Time
  time_plot <- ggplot(time_df, aes(x = SampleSize, y = Time, color = Method, linetype = Method)) +
    geom_point() +
    geom_line() +
    scale_x_log10(breaks = sample_sizes, labels = sample_sizes) +  # Logarithmic scale for x-axis
    scale_y_log10() +  # Logarithmic scale for y-axis
    scale_color_manual(values = colors) +  # Assign colors
    scale_linetype_manual(values = line_types) +  # Assign line types
    labs(title = "Execution Time vs Sample Size (Log Scale)",
         x = "Sample Size (Log Scale)",
         y = "Execution Time (Log Scale)",
         color = "Method") +  # Add legend title
    theme_minimal() +
    theme(legend.position = "right")  # Ensure legend is visible

  # Chemin du dossier où vous souhaitez enregistrer les fichiers
  dossier <- "./Figures"

  # Créer le dossier s'il n'existe pas
  if (!dir.exists(dossier)) {
    dir.create(dossier, showWarnings = FALSE)
  }

  # Save joy plots
  file_path_fig_pehe <- paste0("./Figures/pehe_plot.pdf")
  file_path_fig_time <- paste0("./Figures/time_plot.pdf")

  ggsave(filename = file_path_fig_pehe, plot = pehe_plot,
         width = 20, height = 12, units = "cm")

  ggsave(filename = file_path_fig_time, plot = time_plot,
         width = 20, height = 12, units = "cm")

  # Return the plots
  return(list(PEHE_Plot = pehe_plot, Time_Plot = time_plot,
                pehe_df=pehe_df, time_df=time_df))
}

# Example usage
sample_sizes <- c(500, 1000, 5000, 10000)  # Example list of sample sizes
plots <- create_plots(sample_sizes)

# Print data frames sorted by method
#cat("pehe_df sorted by Method:\n")
#print(plots$pehe_df %>% arrange(SampleSize))

#cat("time_df sorted by Method:\n")
#print(plots$time_df %>% arrange(Method))
