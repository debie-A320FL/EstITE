###########################
# PEHE Distribution Plots #
###########################

rm(list=ls())

# Libraries
library(tidyverse)

# Read Single PEHE file 
curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./../..')


# PEHE_Train = cbind(read.csv("./ACTG/Results/ACTG_1000_CATT_Train_PEHE.csv"), 
#                    read.csv("./ACTG/Results/GP_1000_CATT_Train_PEHE.csv"))
# PEHE_Test = cbind(read.csv("./ACTG/Results/ACTG_1000_CATT_Test_PEHE.csv"), 
#                   read.csv("./ACTG/Results/GP_1000_CATT_Test_PEHE.csv"))

# Define the variable - number of iteration in the simulation
num_ACTG <- 80
num_GP <- 80

# Construct the file paths dynamically
file_path_train_actg <- paste0("./Results/Logit_", num_ACTG, "_CATT_Train_PEHE.csv")
file_path_train_gp <- paste0("./Results/GP_", num_GP, "_CATT_Train_PEHE.csv")
file_path_test_actg <- paste0("./Results/Logit_", num_ACTG, "_CATT_Test_PEHE.csv")
file_path_test_gp <- paste0("./Results/GP_", num_GP, "_CATT_Test_PEHE.csv")

# Read the CSV files using the constructed file paths
PEHE_Train <- cbind(read.csv(file_path_train_actg), read.csv(file_path_train_gp))
PEHE_Test <- cbind(read.csv(file_path_test_actg), read.csv(file_path_test_gp))


PEHE_Train$X = NULL; PEHE_Test$X = NULL

PEHE_Train = reshape(data = PEHE_Train, varying = list(names(PEHE_Train)), timevar = "Model", 
                 times = names(PEHE_Train), direction = "long", v.names = "PEHE")
PEHE_Test = reshape(data = PEHE_Test, varying = list(names(PEHE_Test)), timevar = "Model", 
                    times = names(PEHE_Test), direction = "long", v.names = "PEHE")

# m_order = c("BCF", "NSGP", "CMGP", "CF", "R-BOOST", "R-LASSO", 
#             "X-BART", "X-RF", "T-BART", "T-RF", "S-BART", "S-RF")

m_order = c("BCF", "NSGP", "CMGP", "CF", "R-BOOST", "R-LASSO", 
            "X-BART", "T-BART", "S-BART")

PEHE_Train = 
  PEHE_Train %>%
  select(-id) %>%
  mutate(Model = gsub("[.]", "-", Model),
         Model = factor(Model,
                        levels = m_order,
                        ordered = TRUE))

PEHE_Test = 
  PEHE_Test %>%
  select(-id) %>%
  mutate(Model = gsub("[.]", "-", Model),
         Model = factor(Model,
                        levels = m_order,
                        ordered = TRUE))

  
# # Plot
# p_train <- 
#   ggplot(PEHE_Train, aes(y = PEHE, x = Model, fill = Model)) + 
#   geom_boxplot(outlier.shape = NA, alpha = 0.6) + scale_y_continuous(limits = c(0, 7), breaks = seq(0, 7, 1)) +
#   geom_hline(yintercept = 0, linetype = "dashed") + ylab(expression(sqrt(PEHE))) +
#   theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
# 
# p_test <- 
#   ggplot(PEHE_Test, aes(y = PEHE, x = Model, fill = Model)) + 
#   geom_boxplot(outlier.shape = NA, alpha = 0.6) + scale_y_continuous(limits = c(0, 7), breaks = seq(0, 7, 1)) +
#   geom_hline(yintercept = 0, linetype = "dashed") + ylab(expression(sqrt(PEHE))) +
#   theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
# 
# p_train
# p_test



# Joy plots
library(ggridges)
library(viridis)

x_scale = 20

joy_train <-
  ggplot(PEHE_Train, aes(y = Model, x = PEHE, fill = 0.5 - abs(0.5 - stat(ecdf)))) + 
  stat_density_ridges(geom = "density_ridges_gradient", calc_ecdf = TRUE, scale = 1.5) +
  scale_fill_viridis_c(name = "Tail Probability", begin = 0.1, direction = -1, option = "C") +
  geom_vline(xintercept = 0, linetype = "dotted") +
  scale_x_continuous(limits = c(0, x_scale), breaks = seq(0, x_scale, 2)) + xlab(expression(sqrt(PEHE))) +
  theme_ridges() + theme(legend.position = "none")


joy_test <-
  ggplot(PEHE_Test, aes(y = Model, x = PEHE, fill = 0.5 - abs(0.5 - stat(ecdf)))) + 
  stat_density_ridges(geom = "density_ridges_gradient", calc_ecdf = TRUE, scale = 1.5) +
  scale_fill_viridis_c(name = "Tail Probability", begin = 0.1, direction = -1, option = "C") +
  geom_vline(xintercept = 0, linetype = "dotted") +
  scale_x_continuous(limits = c(0, x_scale), breaks = seq(0, x_scale, 2)) + xlab(expression(sqrt(PEHE))) +
  theme_ridges() + theme(legend.position = "none")


# Chemin du dossier où vous souhaitez enregistrer les fichiers
dossier <- "./Figures"

# Créer le dossier s'il n'existe pas
if (!dir.exists(dossier)) {
  dir.create(dossier, showWarnings = FALSE)
}

# Save joy plots
ggsave(filename = "./Figures/ACTG_Joy_Train.pdf", plot = joy_train, 
       width = 14, height = 12, units = "cm")

ggsave(filename = "./Figures/ACTG_Joy_Test.pdf", plot = joy_test, 
       width = 14, height = 12, units = "cm")
