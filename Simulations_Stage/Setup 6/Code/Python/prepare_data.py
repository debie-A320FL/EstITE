import numpy as np
import pandas as pd
import os

def generate_data(n=1000,
                  age_mean=48, age_sd=6,
                  weight_mean_male=80, weight_sd_male=8,
                  weight_mean_female=65, weight_sd_female=6,
                  comorbidities_prob=0.3,
                  beta_0=-3.5, beta_1=0.05, beta_2=0.02, beta_3=0.5, beta_4=0.3,
                  gamma_0=0.1, gamma_1=0.01, gamma_2=0.005, gamma_3=0.2, gamma_4=0.1,
                  delta_0=-0.1, delta_1=0.005, delta_2=0.002, delta_3=0.1, delta_4=0.05,
                  sigma_sq=0.1, seed=123,export =False):
    np.random.seed(seed)

    age = np.random.normal(age_mean, age_sd, n)
    gender = np.random.binomial(1, 0.5, n)
    weight = np.where(gender == 0,
                      np.random.normal(weight_mean_male, weight_sd_male, n),
                      np.random.normal(weight_mean_female, weight_sd_female, n))
    comorbidities = np.random.binomial(1, comorbidities_prob, n)

    prob_treatment = 1 / (1 + np.exp(-(beta_0 + beta_1 * age + beta_2 * weight + beta_3 * comorbidities + beta_4 * gender)))
    treatment = np.random.binomial(1, prob_treatment)

    mu_0 = gamma_0 + gamma_1 * age + gamma_2 * weight + gamma_3 * comorbidities + gamma_4 * gender
    tau = delta_0 + delta_1 * age + delta_2 * weight + delta_3 * comorbidities + delta_4 * gender

    epsilon = np.random.normal(0, np.sqrt(sigma_sq), n)
    prob_Y = 1 / (1 + np.exp(-(mu_0 + treatment * tau + epsilon)))
    Y = np.random.binomial(1, prob_Y)

    data = pd.DataFrame({
        'age': np.round(age).astype(int),
        'weight': np.round(weight, 1),
        'gender': gender.astype(int),
        'comorbidities': comorbidities.astype(int),
        'treatment': treatment.astype(int),
        'Y': Y.astype(int)
    })

    hyperparams = pd.DataFrame({
        'beta_0': [beta_0], 'beta_1': [beta_1], 'beta_2': [beta_2], 'beta_3': [beta_3], 'beta_4': [beta_4],
        'gamma_0': [gamma_0], 'gamma_1': [gamma_1], 'gamma_2': [gamma_2], 'gamma_3': [gamma_3], 'gamma_4': [gamma_4],
        'delta_0': [delta_0], 'delta_1': [delta_1], 'delta_2': [delta_2], 'delta_3': [delta_3], 'delta_4': [delta_4],
        'sigma_sq': [sigma_sq]
    })
    if export:
        hyperparams.to_csv("hyperparams.csv", index=False)

    return data

def export_data_to_csv(data, file_name="simulated_data.csv", directory=".", overwrite=False):
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")

    if os.path.exists(file_path):
        if overwrite:
            data.to_csv(file_path, index=False)
            print(f"Existing file overwritten and data exported to {file_path}")
        else:
            print(f"File already exists and overwrite is False. Data not exported.")
    else:
        data.to_csv(file_path, index=False)
        print(f"Data exported to {file_path}")

def find_optimal_beta_0(target_proportion, n=int(1e6), max_iter=100):
    if not (0 < target_proportion < 1):
        raise ValueError("target_proportion should be strictly between 0 and 1")

    beta_0_low = -1
    beta_0_high = 1
    tolerance = target_proportion / 100

    while True:
        data_low = generate_data(n=n, beta_0=beta_0_low)
        proportion_low = (data_low['treatment'] == 0).mean()

        data_high = generate_data(n=n, beta_0=beta_0_high)
        proportion_high = (data_high['treatment'] == 0).mean()

        if proportion_low > target_proportion and proportion_high < target_proportion:
            break
        elif proportion_low > target_proportion:
            beta_0_high *= 2
        else:
            beta_0_low *= 2

    iter_count = 0
    while beta_0_high - beta_0_low > tolerance and iter_count < max_iter:
        beta_0_mid = (beta_0_low + beta_0_high) / 2
        data = generate_data(n=n, beta_0=beta_0_mid)
        current_proportion = (data['treatment'] == 0).mean()

        if abs(current_proportion - target_proportion) < tolerance:
            break
        elif current_proportion < target_proportion:
            beta_0_high = beta_0_mid
        else:
            beta_0_low = beta_0_mid

        iter_count += 1
    
    print(f"iter_count = {iter_count}")

    return beta_0_mid

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from scipy.special import expit as logistic
import random

def prepare_train_data(size_sample, hyperparams, seed=123, train_ratio=0.7,
                       treatment_percentile=35, verbose=False, binary=False):

    np.random.seed(seed)
    random.seed(seed)

    hyperparams["beta_0"] = find_optimal_beta_0(treatment_percentile/100)
    data = generate_data(n=size_sample,  **hyperparams)
    myX = data.drop(columns=["treatment", "Y"])
    myY = data["Y"]
    myZ = data["treatment"]
    if binary:
        myY = np.random.binomial(1, myY)

    # Train-test split
    N = size_sample
    mysplit = [1] * int(train_ratio * N) + [2] * (N - int(train_ratio * N))
    random.shuffle(mysplit)

    smp_split = np.array(mysplit)

    y_train = myY[smp_split == 1]
    y_test = myY[smp_split == 2]

    x_train = myX[smp_split == 1]
    x_test = myX[smp_split == 2]

    z_train = myZ[smp_split == 1]
    z_test = myZ[smp_split == 2]

    # Calculer mu_0, tau, et ITE
    column_names = ["age", "weight", "comorbidities", "gender"]
    column_indices = [data.columns.get_loc(col) for col in column_names]

    mu_0 = (hyperparams['gamma_0'] * np.ones(myX.shape[0]) +
            hyperparams['gamma_1'] * myX["age"] +  # age
            hyperparams['gamma_2'] * myX["weight"] +  # weight
            hyperparams['gamma_3'] * myX["comorbidities"] +  # comorbidities
            hyperparams['gamma_4'] * myX["gender"])   # gender

    tau = (hyperparams['delta_0'] * np.ones(myX.shape[0]) +
        hyperparams['delta_1'] * myX["age"] +  # age
        hyperparams['delta_2'] * myX["weight"] +  # weight
        hyperparams['delta_3'] * myX["comorbidities"] +  # comorbidities
        hyperparams['delta_4'] * myX["gender"])   # gender

    ITE_proba = 1 / (1 + np.exp(-(mu_0 + tau))) - 1 / (1 + np.exp(-mu_0))

    Train_ITE = ITE_proba[smp_split == 1]
    Test_ITE = ITE_proba[smp_split == 2]

    Train_CATT = Train_ITE[z_train == 1]
    Train_CATC = Train_ITE[z_train == 0]

    Test_CATT = Test_ITE[z_test == 1]
    Test_CATC = Test_ITE[z_test == 0]

    return {
        "x_train": x_train, "z_train": z_train, "y_train": y_train,
        "x_test": x_test, "z_test": z_test, "y_test": y_test,
        "Test_CATT": Test_CATT, "Test_CATC": Test_CATC, "Test_ITE" :Test_ITE,
    }


if __name__ == "__main__":
    import pandas as pd

    import os

    print("hello")
    os.chdir("/home/onyxia/work/EstITE/Simulations_Stage/Setup 6")
    hyperparams_df = pd.read_csv("./../Setup 1a/Data/hyperparams.csv")
    column_names = hyperparams_df.columns.tolist()

    # Convert first row to dict
    hyperparams_dict = hyperparams_df.iloc[0].to_dict()
    print(column_names)
    hyperparams_dict["beta_0"] = find_optimal_beta_0(5/100)
    data = generate_data(n=1000,  **hyperparams_dict)
    mean_Y = data["Y"].mean()
    print(f"Mean of Y: {mean_Y}")

    mean_Z = data["treatment"].mean()
    print(f"Mean of Z: {mean_Z}")

    if 0:

        os.chdir("/home/onyxia/work/EstITE/Simulations_Stage/Setup 5c")

        # Load data
        hyperparams_df = pd.read_csv("./../Setup 1a/Data/hyperparams.csv")
        data_train_test = pd.read_csv("./../Setup 1a/Data/simulated_1M_data.csv")

        data_validation = pd.read_csv("./../Setup 1a/Data/simulated_10K_data_validation.csv")
        size_sample_val = data_validation.shape[0]

        # Convert hyperparams to dictionary (if it's a single row)
        hyperparams = hyperparams_df.iloc[0].to_dict()
        treatment_percentile = 10
        binary = True

        # Prepare validation data
        res_val = prepare_train_data(
            data=data_validation,
            hyperparams=hyperparams,
            size_sample=size_sample_val,
            train_ratio=0,
            treatment_percentile=treatment_percentile,
            verbose=True,
            binary=binary
        )

        # Unpack results
        val_augmX = res_val['test_augmX']
        z_val     = res_val['z_test']
        y_val     = res_val['y_test']
        val_CATT  = res_val['Test_CATT']