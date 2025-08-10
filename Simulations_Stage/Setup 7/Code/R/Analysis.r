

setwd("/home/onyxia/work/EstITE/Simulations_Stage/Setup 7")

if(!require(stringr))     install.packages("stringr")

library(stringr)


summary_stats <- function(csv_path, export = FALSE, pdf_folder = "output") {
  # Load packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
  if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")
  if (!requireNamespace("ggridges", quietly = TRUE)) install.packages("ggridges")
  if (!requireNamespace("viridis", quietly = TRUE)) install.packages("viridis")
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
  
  library(ggplot2)
  library(tidyr)
  library(ggridges)
  library(viridis)
  library(dplyr)
  
  # Read CSV
  data <- read.csv(csv_path, stringsAsFactors = FALSE)
  
  # Remove first column if it's simulation index
  if (is.numeric(data[[1]]) && all(data[[1]] %% 1 == 0)) {
    data <- data[ , -1]
  }
  
  # Compute summary stats table
  compute_stats <- function(x) {
    c(
      D1     = quantile(x, 0.10, na.rm = TRUE),
      Q1     = quantile(x, 0.25, na.rm = TRUE),
      Median = median(x, na.rm = TRUE),
      Mean   = mean(x, na.rm = TRUE),
      Q3     = quantile(x, 0.75, na.rm = TRUE),
      D9     = quantile(x, 0.90, na.rm = TRUE)
    )
  }
  
  stats_df <- as.data.frame(t(sapply(data, compute_stats)))
  print(round(stats_df, 4))
  
  if (export) {
    # Ensure folder exists
    if (!dir.exists(pdf_folder)) dir.create(pdf_folder, recursive = TRUE)
    
    # Clean filename
    file_base <- basename(csv_path)
    file_base <- sub("^pehe_.*?_", "", file_base) # remove prefix like pehe_B_500_
    file_base <- sub("\\.csv$", "", file_base)    # remove extension
    
    pdf_file <- file.path(pdf_folder, paste0("distribution_ridgeline_", file_base, ".pdf"))
    
    # Long format
    data_long <- pivot_longer(data, cols = everything(), names_to = "Model", values_to = "PEHE")
    print(unique(data_long$Model))
    data_long <- data_long %>% filter(Model != "Zero.learner")
    print(unique(data_long$Model))

    
    # Order by median
    medians <- aggregate(PEHE ~ Model, data_long, median)
    model_order <- medians$Model[order(medians$PEHE)]
    data_long$Model <- factor(data_long$Model, levels = model_order, ordered = TRUE)
    
    # Compute stats for plotting using dplyr
    stats_plot <- data_long %>%
      group_by(Model) %>%
      summarise(
        D1     = quantile(PEHE, 0.10, na.rm = TRUE),
        Median = median(PEHE, na.rm = TRUE),
        Mean   = mean(PEHE, na.rm = TRUE),
        D9     = quantile(PEHE, 0.90, na.rm = TRUE),
        .groups = "drop"
      )
    
    # Global 2nd & 98th percentile for axis limits
    global_min <- quantile(data_long$PEHE, 0.001, na.rm = TRUE)
    global_max <- quantile(data_long$PEHE, 0.999, na.rm = TRUE)
    
    # Ridgeline plot
    p <- ggplot(data_long, aes(y = Model, x = PEHE,
                            fill = after_stat(0.5 - abs(0.5 - ecdf)))) +
    stat_density_ridges(geom = "density_ridges_gradient", calc_ecdf = TRUE,
                        scale = 1.5, alpha = 0.8) +
    scale_fill_viridis_c(name = "Tail Probability", begin = 0.1,
                        direction = -1, option = "C") +
    geom_vline(xintercept = 0, linetype = "dotted") +
    # Add mean & quantile lines
    geom_segment(data = stats_plot, aes(x = Mean, xend = Mean,
                                        y = as.numeric(Model) - 0.45,
                                        yend = as.numeric(Model) + 0.45),
                inherit.aes = FALSE, color = "black", size = 0.8) +
    geom_segment(data = stats_plot, aes(x = D1, xend = D1,
                                        y = as.numeric(Model) - 0.45,
                                        yend = as.numeric(Model) + 0.45),
                inherit.aes = FALSE, color = "red", linetype = "dashed", size = 0.6) +
    geom_segment(data = stats_plot, aes(x = D9, xend = D9,
                                        y = as.numeric(Model) - 0.45,
                                        yend = as.numeric(Model) + 0.45),
                inherit.aes = FALSE, color = "red", linetype = "dashed", size = 0.6) +
    geom_segment(data = stats_plot, aes(x = Median, xend = Median,
                                        y = as.numeric(Model) - 0.45,
                                        yend = as.numeric(Model) + 0.45),
                inherit.aes = FALSE, color = "blue", linetype = "dotted", size = 0.6) +
    scale_x_continuous(limits = c(global_min, global_max)) +
    xlab(expression(sqrt(PEHE))) +
    theme_ridges() +
    theme(legend.position = "none")
        
    # Save PDF
    pdf(pdf_file, width = 8, height = 6)
    print(p)
    dev.off()
    
    message("Ridgeline PDF exported to: ", pdf_file)
  }
}


# Example usage:

path = "Results/pehe_B_500_x_dim_25_N_100000_scenario_4_sigma_0.3.csv"
summary_stats(path, export = TRUE)
