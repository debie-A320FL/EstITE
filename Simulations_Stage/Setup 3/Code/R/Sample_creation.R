# Charger les bibliothèques nécessaires
library(dplyr)

curr_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(curr_dir); setwd('./../../Data')


# Fonction pour générer les données
generate_data <- function(n = 1000,
                          age_mean = 48, age_sd = 6,
                          weight_mean_male = 80, weight_sd_male = 8,
                          weight_mean_female = 65, weight_sd_female = 6,
                          comorbidities_prob = 0.3,
                          beta_0 = -3.5, beta_1 = 0.05, beta_2 = 0.02, beta_3 = 0.5, beta_4 = 0.3,
                          gamma_0 = 0.1, gamma_1 = 0.01, gamma_2 = 0.005, gamma_3 = 0.2, gamma_4 = 0.1,
                          delta_0 = -0.1, delta_1 = 0.005, delta_2 = 0.002, delta_3 = 0.1, delta_4 = 0.05,
                          sigma_sq = 0.1) {
  # Documentation des paramètres
  # n : Nombre d'individus à générer
  # age_mean, age_sd : Moyenne et écart-type de la distribution de l'âge
  # weight_mean_male, weight_sd_male : Moyenne et écart-type de la distribution du poids pour les hommes
  # weight_mean_female, weight_sd_female : Moyenne et écart-type de la distribution du poids pour les femmes
  # comorbidities_prob : Probabilité de présence de comorbidités
  # beta_0, beta_1, beta_2, beta_3, beta_4 : Coefficients du modèle de propension pour le traitement
  # gamma_0, gamma_1, gamma_2, gamma_3, gamma_4 : Coefficients de la fonction de base (baseline function)
  # delta_0, delta_1, delta_2, delta_3, delta_4 : Coefficients de l'effet causal conditionnel (CATE)
  # sigma_sq : Variance du terme d'erreur
  
  # Équations utilisées dans la simulation :
  # 1. Modèle de propension pour le traitement :
  #    P(T_i = 1 | X_i) = 1 / (1 + exp(-(beta_0 + beta_1 * Age_i + beta_2 * Poids_i + beta_3 * Comorbidités_i + beta_4 * Genre_i)))
  # 2. Fonction de base (baseline function) :
  #    mu_0(X_i) = gamma_0 + gamma_1 * Age_i + gamma_2 * Poids_i + gamma_3 * Comorbidités_i + gamma_4 * Genre_i
  # 3. Effet causal conditionnel (CATE) :
  #    tau(X_i) = delta_0 + delta_1 * Age_i + delta_2 * Poids_i + delta_3 * Comorbidités_i + delta_4 * Genre_i
  # 4. Issue Y :
  #    Y_i ~ Bernoulli(p_i) où p_i = 1 / (1 + exp(-(mu_0(X_i) + T_i * tau(X_i) + epsilon_i)))
  
  # Générer les caractéristiques des individus
  set.seed(123) # Pour la reproductibilité
  age <- rnorm(n, mean = age_mean, sd = age_sd)
  gender <- rbinom(n, 1, 0.5) # 0 pour homme, 1 pour femme
  weight <- ifelse(gender == 0,
                   rnorm(n, mean = weight_mean_male, sd = weight_sd_male),
                   rnorm(n, mean = weight_mean_female, sd = weight_sd_female))
  comorbidities <- rbinom(n, 1, comorbidities_prob)
  
  # Calculer la probabilité de traitement
  prob_treatment <- 1 / (1 + exp(-(beta_0 + beta_1 * age + beta_2 * weight + beta_3 * comorbidities + beta_4 * gender)))
  treatment <- rbinom(n, 1, prob_treatment)
  
  # Calculer la fonction de base et l'effet causal conditionnel
  mu_0 <- gamma_0 + gamma_1 * age + gamma_2 * weight + gamma_3 * comorbidities + gamma_4 * gender
  tau <- delta_0 + delta_1 * age + delta_2 * weight + delta_3 * comorbidities + delta_4 * gender
  
  # Générer le terme d'erreur
  epsilon <- rnorm(n, mean = 0, sd = sqrt(sigma_sq))
  
  # Calculer la probabilité de l'issue Y
  prob_Y <- 1 / (1 + exp(-(mu_0 + treatment * tau + epsilon)))
  
  # Tirer l'issue Y à partir de la probabilité
  Y <- rbinom(n, 1, prob_Y)
  
  # Créer un data frame avec les données générées
  data <- data.frame(age = round(age), weight = round(weight,1), gender = gender, comorbidities = comorbidities, treatment = treatment, Y = Y)
  
  # Convertir les colonnes gender, treatment et Y en facteurs
  data <- data %>%
    mutate(gender = factor(gender, levels = c(0, 1), labels = c(0, 1)),
           treatment = factor(treatment, levels = c(0, 1), labels = c(0, 1)),
           Y = factor(Y, levels = c(0, 1), labels = c(0, 1)),
           comorbidities = factor(comorbidities, level = c(0,1), labels = c(0, 1)))
  
  # Exporter les hyperparamètres dans un fichier CSV
  hyperparams <- data.frame(
    beta_0 = beta_0, beta_1 = beta_1, beta_2 = beta_2, beta_3 = beta_3, beta_4 = beta_4,
    gamma_0 = gamma_0, gamma_1 = gamma_1, gamma_2 = gamma_2, gamma_3 = gamma_3, gamma_4 = gamma_4,
    delta_0 = delta_0, delta_1 = delta_1, delta_2 = delta_2, delta_3 = delta_3, delta_4 = delta_4,
    sigma_sq = sigma_sq
  )
  write.csv(hyperparams, "hyperparams.csv", row.names = FALSE)
  
  return(data)
}


# Fonction pour exporter les données dans un fichier CSV
export_data_to_csv <- function(data, file_name = "simulated_data.csv", directory = ".", overwrite = FALSE) {
  # Créer le chemin complet du fichier
  file_path <- file.path(directory, file_name)
  
  # Vérifier si le dossier existe, sinon le créer
  if (!dir.exists(directory)) {
    dir.create(directory, recursive = TRUE)
    cat("Le dossier", directory, "a été créé.\n")
  }
  
  # Vérifier si le fichier existe déjà
  if (file.exists(file_path)) {
    if (overwrite) {
      write.csv(data, file_path, row.names = FALSE)
      cat("Le fichier existant a été écrasé et les nouvelles données ont été exportées dans le fichier", file_path, "\n")
    } else {
      cat("Le fichier existe déjà et overwrite est FALSE. Les données n'ont pas été exportées.\n")
    }
  } else {
    write.csv(data, file_path, row.names = FALSE)
    cat("Les données ont été exportées dans le fichier", file_path, "\n")
  }
}

data <- generate_data(n = 1000000)

data %>% head()


data %>% summary()

# Afficher le résumé des données pour les hommes (gender == 0)
summary_males <- data %>% filter(gender == 0) %>% select(-gender) %>% summary()
print(summary_males)

# Afficher le résumé des données pour les femmes (gender == 0)
summary_females <- data %>% filter(gender == 1) %>% select(-gender) %>% summary()
print(summary_females)

data %>%
  filter(gender == 0) %>%
  summarise(proportion = mean(treatment == 1))

export_data_to_csv(data, file_name = "simulated_1M_data.csv",
                   directory = ".",
                   overwrite = FALSE)
