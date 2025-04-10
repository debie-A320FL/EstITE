# Importing packages
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

from sklearn import preprocessing
from scipy import stats as sts

# Ajouter le chemin du dossier Python de Setup 1 à sys.path
import sys

basedir_setup_1 = "/home/onyxia/work/EstITE/Simulations_Stage/Setup 1/Data"
data = pd.read_csv(basedir_setup_1 + "/simulated_1M_data.csv")

size_sample = 1000
data = data.sample(n=size_sample, random_state=1)  # You can set a random_state for reproducibility

# Define treatment assignment
myZ = np.array(data["treatment"])
myX = np.array(data.drop(columns=["treatment"]))

# Importer les hyperparamètres
hyperparams = pd.read_csv(basedir_setup_1 + "/hyperparams.csv")

# Obtenir les indices des colonnes par leurs noms
column_names = ["age", "weight", "comorbidities", "gender"]
column_indices = [data.columns.get_loc(col) for col in column_names]

# Calculer mu_0, tau, et ITE
mu_0 = (hyperparams['gamma_0'].values[0] * np.ones(myX.shape[0]) +
        hyperparams['gamma_1'].values[0] * myX[:, column_indices[0]] +  # age
        hyperparams['gamma_2'].values[0] * myX[:, column_indices[1]] +  # weight
        hyperparams['gamma_3'].values[0] * myX[:, column_indices[2]] +  # comorbidities
        hyperparams['gamma_4'].values[0] * myX[:, column_indices[3]])   # gender

tau = (hyperparams['delta_0'].values[0] * np.ones(myX.shape[0]) +
       hyperparams['delta_1'].values[0] * myX[:, column_indices[0]] +  # age
       hyperparams['delta_2'].values[0] * myX[:, column_indices[1]] +  # weight
       hyperparams['delta_3'].values[0] * myX[:, column_indices[2]] +  # comorbidities
       hyperparams['delta_4'].values[0] * myX[:, column_indices[3]])   # gender

ITE = mu_0 + tau * myZ
# Générer le vecteur de bruit gaussien
# bruit_gaussien = np.random.normal(0, hyperparams['sigma_sq'], size_sample)

# Fonction logistique (logit inverse)
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Générer le vecteur de bruit gaussien
bruit_gaussien = np.random.normal(0, hyperparams['sigma_sq'], size_sample)
#bruit_gaussien = 0
data["Y_proba_b"] = logistic(ITE + bruit_gaussien * 10)
data["Y_proba"] = logistic(ITE)

import matplotlib.pyplot as plt


# Trier les données par Y_proba
data_sorted = data.sort_values(by="Y_proba")

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculer la MSE et la MAE
mse = mean_squared_error(data_sorted["Y_proba"], data_sorted["Y_proba_b"])
mae = mean_absolute_error(data_sorted["Y_proba"], data_sorted["Y_proba_b"])

print(f"MSE: {mse}")
print(f"MAE: {mae}")

# Calculer la différence entre Y_proba et Y_proba_b
difference = data_sorted["Y_proba"] - data_sorted["Y_proba_b"]

# Calculer les déciles de la différence
deciles_difference = difference.quantile(np.arange(0.1, 1.1, 0.1))

print("Déciles de la différence entre Y_proba et Y_proba_b:")
print(deciles_difference)