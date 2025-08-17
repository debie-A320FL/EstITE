import numpy as np
import pandas as pd
import os

from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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


def prepare_train_data_null_cate_indep_treatment(size_sample, x_dim, seed=123,
                                                 train_ratio=0.7, treatment_prob=0.35,
                                                 binary=False):
    np.random.seed(seed)
    random.seed(seed)

    # Generate independent Gaussian features
    X = np.random.normal(0, 1, size=(size_sample, x_dim))
    feature_names = [f"x{i}" for i in range(x_dim)]
    myX = pd.DataFrame(X, columns=feature_names)

    # Treatment assignment: independent Bernoulli
    myZ = np.random.binomial(1, treatment_prob, size=size_sample)

    # Outcome generation: only depends on X
    gamma = np.random.uniform(-1, 1, x_dim)
    mu = myX.values @ gamma
    prob_y = 1 / (1 + np.exp(-mu))  # probability of positive outcome

    myY = prob_y
    if binary:
        myY = np.random.binomial(1, prob_y)

    # Train-test split
    N = size_sample
    mysplit = [1] * int(train_ratio * N) + [2] * (N - int(train_ratio * N))
    random.shuffle(mysplit)
    smp_split = np.array(mysplit)

    x_train = myX[smp_split == 1]
    x_test = myX[smp_split == 2]
    z_train = myZ[smp_split == 1]
    z_test = myZ[smp_split == 2]
    y_train = myY[smp_split == 1]
    y_test = myY[smp_split == 2]

    # True ITEs are zero
    ITE = np.zeros(size_sample)
    Train_ITE = ITE[smp_split == 1]
    Test_ITE = ITE[smp_split == 2]

    Test_CATT = Test_ITE[z_test == 1]
    Test_CATC = Test_ITE[z_test == 0]

    return {
        "x_train": x_train, "z_train": z_train, "y_train": y_train,
        "x_test": x_test, "z_test": z_test, "y_test": y_test,
        "Test_CATT": Test_CATT, "Test_CATC": Test_CATC, "Test_ITE": Test_ITE,
    }

def prepare_train_data_scenario1(
    size_sample: int,
    x_dim: int,
    k_mu: int = 5,              # Number of features used in outcome model μ₀
    k_conf: int = 5,            # Additional features influencing treatment (confounders)
    non_treated_frac: float = 0.1,  # Desired fraction of untreated individuals
    seed: int = 123,
    train_ratio: float = 0.7,
    noise_std: float = 1.0,
    binary: bool = False
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    # 1. Generate Gaussian features
    X = np.random.normal(0, 1, size=(size_sample, x_dim))
    feature_names = [f"x{j}" for j in range(x_dim)]
    dfX = pd.DataFrame(X, columns=feature_names)

    # 2. Randomly select disjoint index sets
    all_indices = list(range(x_dim))
    random.shuffle(all_indices)
    S_mu = all_indices[:k_mu]
    S_conf = all_indices[k_mu:k_mu + k_conf]
    S_pi = S_mu + S_conf  # Features influencing treatment assignment

    # 3. Sample coefficients
    gamma0 = 0.0
    gamma = np.random.normal(0, 1, x_dim)  # for μ₀
    alpha0 = 0.0
    alpha = np.random.normal(0, 1, x_dim)  # for π(x)

    # 4. Compute μ₀(x)
    mu0 = gamma0 + X[:, S_mu] @ gamma[S_mu]

    # 5. τ(x) = 0 → null CATE
    tau = np.zeros(size_sample)

    # 6. Compute π(x) = logistic(α₀ + αᵗx)
    pi_logits = alpha0 + X[:, S_pi] @ alpha[S_pi]
    pi = 1 / (1 + np.exp(-pi_logits))

    # 7. Assign treatment to match desired untreated proportion
    n_control = int(non_treated_frac * size_sample)
    n_treated = size_sample - n_control

    # Sort by π(x) descending and assign treatment
    sorted_indices = np.argsort(-pi)
    Z = np.zeros(size_sample, dtype=int)
    Z[sorted_indices[:n_treated]] = 1  # Top individuals more likely to be treated

    # 8. Generate outcome Y = μ₀(x) + ε
    Y_cont = mu0 + np.random.normal(0, noise_std, size=size_sample)

    if binary:
        pY = 1 / (1 + np.exp(-Y_cont))
        Y = np.random.binomial(1, pY)
    else:
        Y = Y_cont

    # 9. Train-test split
    indices = np.arange(size_sample)
    np.random.shuffle(indices)
    train_size = int(train_ratio * size_sample)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    def split(arr):
        return arr[train_idx], arr[test_idx]

    x_train = dfX.iloc[train_idx]
    x_test = dfX.iloc[test_idx]
    z_train, z_test = split(Z)
    y_train, y_test = split(Y)

    ITE_test = np.zeros(len(test_idx))
    Test_CATT = ITE_test[z_test == 1]
    Test_CATC = ITE_test[z_test == 0]

    return {
        "x_train": x_train, "z_train": z_train, "y_train": y_train,
        "x_test": x_test,   "z_test": z_test,   "y_test": y_test,
        "Test_ITE": ITE_test, "Test_CATT": Test_CATT, "Test_CATC": Test_CATC,
    }

def prepare_train_data_scenario2(
    size_sample: int,
    x_dim: int,
    k_mu: int = 5,            # Number of features used in outcome model μ₀
    k_conf: int = 5,          # Additional features used in π(x)
    k_tau: int = 5,           # Features used in τ(x)
    seed: int = 123,
    train_ratio: float = 0.7,
    noise_std: float = 1.0,
    binary: bool = False,
    non_treated_frac: float = None  # e.g., 0.1 for forcing ~10% untreated
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    # 1. Generate Gaussian features
    X = np.random.normal(0, 1, size=(size_sample, x_dim))
    feature_names = [f"x{j}" for j in range(x_dim)]
    dfX = pd.DataFrame(X, columns=feature_names)

    # 2. Randomly select disjoint index sets
    all_indices = list(range(x_dim))
    random.shuffle(all_indices)
    S_mu = all_indices[:k_mu]
    S_conf = all_indices[k_mu:k_mu + k_conf]
    S_tau = all_indices[k_mu + k_conf:k_mu + k_conf + k_tau]
    S_pi = S_mu + S_conf  # features influencing treatment assignment

    # 3. Sample coefficients
    gamma0 = 0.0
    gamma = np.random.normal(0, 1, x_dim)  # for μ₀(x)

    alpha0 = 0.0
    alpha = np.random.normal(0, 1, x_dim)  # for π(x)

    delta0 = 0.0
    delta = np.random.normal(0, 1, x_dim)  # for τ(x)

    # 4. Compute μ₀(x)
    mu0 = gamma0 + X[:, S_mu] @ gamma[S_mu]

    # 5. Compute τ(x)
    tau = delta0 + X[:, S_tau] @ delta[S_tau]

    # 6. Compute π(x)
    pi_logits = alpha0 + X[:, S_pi] @ alpha[S_pi]
    pi = 1 / (1 + np.exp(-pi_logits))

    # 7. Sample treatment Z ~ Bern(pi)
    if non_treated_frac is not None:
        n_control = int(non_treated_frac * size_sample)
        n_treated = size_sample - n_control
        sorted_indices = np.argsort(-pi)  # sort by decreasing pi(x)
        Z = np.zeros(size_sample, dtype=int)
        Z[sorted_indices[:n_treated]] = 1
    else:
        Z = np.random.binomial(1, pi)

    # 8. Generate potential outcomes
    Y0 = mu0 + np.random.normal(0, noise_std, size=size_sample)
    Y1 = mu0 + tau + np.random.normal(0, noise_std, size=size_sample)

    # Observed outcome
    Y_cont = Y0 * (1 - Z) + Y1 * Z

    # Optional binarization
    if binary:
        pY = 1 / (1 + np.exp(-Y_cont))
        Y = np.random.binomial(1, pY)
        tau = 1 / (1 + np.exp(-Y1)) - 1 / (1 + np.exp(-Y0))
    else:
        Y = Y_cont

    # 9. Train-test split
    N = size_sample
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_size = int(train_ratio * N)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    def split(arr):
        return arr[train_idx], arr[test_idx]

    x_train = dfX.iloc[train_idx]
    x_test = dfX.iloc[test_idx]
    z_train, z_test = split(Z)
    y_train, y_test = split(Y)
    ITE = tau  # τ(x) is the true ITE

    ITE_train, ITE_test = split(ITE)
    Test_CATT = ITE_test[z_test == 1]
    Test_CATC = ITE_test[z_test == 0]

    return {
        "x_train": x_train,   "z_train": z_train,   "y_train": y_train,
        "x_test": x_test,     "z_test": z_test,     "y_test": y_test,
        "Test_ITE": ITE_test, "Test_CATT": Test_CATT, "Test_CATC": Test_CATC,
    }

def prepare_train_data_scenario3(
    size_sample: int,
    x_dim: int,
    k_mu: int = 5,             # Number of features for μ₀(x)
    k_tau: int = 5,            # Number of features for μ₁(x)
    k_pi: int = 5,             # Features affecting π(x)
    seed: int = 123,
    train_ratio: float = 0.7,
    noise_std: float = 1.0,
    binary: bool = False,
    non_treated_frac: float = None  # e.g., 0.1 for forcing 10% untreated
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    # 1. Generate Gaussian covariates
    X = np.random.normal(0, 1, size=(size_sample, x_dim))
    feature_names = [f"x{j}" for j in range(x_dim)]
    dfX = pd.DataFrame(X, columns=feature_names)

    # 2. Select disjoint subsets of features
    all_indices = list(range(x_dim))
    random.shuffle(all_indices)
    S_0 = all_indices[:k_mu]
    S_1 = all_indices[k_mu:k_mu + k_tau]
    S_pi = all_indices[k_mu + k_tau:k_mu + k_tau + k_pi]

    # 3. Sample coefficients
    gamma0 = 0.0
    gamma = np.random.normal(0, 1, x_dim)  # for μ₀(x)
    beta0 = 0.0
    beta = np.random.normal(0, 1, x_dim)   # for μ₁(x)
    alpha0 = 0.0
    alpha = np.random.normal(0, 1, x_dim)  # for π(x)

    # 4. Compute μ₀(x) and μ₁(x)
    mu0 = gamma0 + X[:, S_0] @ gamma[S_0]
    mu1 = beta0 + X[:, S_1] @ beta[S_1]

    # True ITE
    tau = mu1 - mu0

    # 5. Compute π(x) = sigmoid(α₀ + αᵗx)
    pi_logits = alpha0 + X[:, S_pi] @ alpha[S_pi]
    pi = 1 / (1 + np.exp(-pi_logits))

    # 6. Treatment assignment
    if non_treated_frac is not None:
        n_control = int(non_treated_frac * size_sample)
        n_treated = size_sample - n_control
        sorted_indices = np.argsort(-pi)
        Z = np.zeros(size_sample, dtype=int)
        Z[sorted_indices[:n_treated]] = 1
    else:
        Z = np.random.binomial(1, pi)

    # 7. Generate potential outcomes
    Y0 = mu0 + np.random.normal(0, noise_std, size=size_sample)
    Y1 = mu1 + np.random.normal(0, noise_std, size=size_sample)

    Y_cont = Y0 * (1 - Z) + Y1 * Z

    # 8. Optional binarization
    if binary:
        pY = 1 / (1 + np.exp(-Y_cont))
        Y = np.random.binomial(1, pY)
        tau = 1 / (1 + np.exp(-Y1)) - 1 / (1 + np.exp(-Y0))
    else:
        Y = Y_cont

    # 9. Train-test split
    indices = np.arange(size_sample)
    np.random.shuffle(indices)
    train_size = int(train_ratio * size_sample)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    def split(arr):
        return arr[train_idx], arr[test_idx]

    x_train = dfX.iloc[train_idx]
    x_test = dfX.iloc[test_idx]
    z_train, z_test = split(Z)
    y_train, y_test = split(Y)
    ITE_train, ITE_test = split(tau)

    Test_CATT = ITE_test[z_test == 1]
    Test_CATC = ITE_test[z_test == 0]

    return {
        "x_train": x_train,   "z_train": z_train,   "y_train": y_train,
        "x_test": x_test,     "z_test": z_test,     "y_test": y_test,
        "Test_ITE": ITE_test, "Test_CATT": Test_CATT, "Test_CATC": Test_CATC,
    }

def make_ar1_cov(rho: float, d: int, sigma: float = 1.0) -> np.ndarray:
    """
    Build a d×d AR(1) covariance matrix with parameter rho and marginal std sigma.
    """
    idx = np.arange(d)
    return sigma * (rho ** np.abs(idx[:, None] - idx[None, :]))

def prepare_train_data_mixture1(
    size_sample: int,
    x_dim: int,
    n_groups: int = 3,
    sigma_list: list[float] = None,
    rho_list: list[float] = None,
    ite_list: list[float] = None,
    prop_list: list[float] = None,
    seed: int = 123,
    train_ratio: float = 0.7,
    noise_std: float = 1.0,
    binary: bool = False,
    non_treated_frac: float = None  # override to force fraction untreated
) -> dict:
    """
    Scenario 1: Gaussian‐mixture covariates + group‐specific constant ITE.

    Parameters
    ----------
    size_sample : int
      total number of samples.
    x_dim : int
      feature dimension.
    n_groups : int, default=3
      number of mixture components / groups.
    sigma_list : list of float, length n_groups
      marginal standard deviations for each component.
    rho_list : list of float, length n_groups
      AR(1) correlation parameters for each component.
    ite_list : list of float, length n_groups
      constant treatment effect for each group.
    prop_list : list of float, length n_groups
      mixture proportions (must sum to 1). Defaults to equal.
    seed : int
      random seed.
    train_ratio : float
      fraction of data to use for training.
    noise_std : float
      noise standard deviation for outcomes.
    binary : bool
      whether to threshold outcomes into binary via logistic.
    non_treated_frac : float, optional
      if set, fixes the fraction of controls to this value.

    Returns
    -------
    dict
      {
        "x_train": DataFrame, "z_train": array, "y_train": array,
        "x_test": DataFrame,  "z_test": array,  "y_test": array,
        "Test_ITE": array,    "Test_CATT": array, "Test_CATC": array
      }
    """
    np.random.seed(seed)

    # Defaults
    if sigma_list is None:
        sigma_list = [1.0] * n_groups
    if rho_list is None:
        rho_list = [0.5] * n_groups
    if ite_list is None:
        ite_list = [1.0 * (i + 1) for i in range(n_groups)]
    if prop_list is None:
        prop_list = [1.0 / n_groups] * n_groups

    # 1) Draw group means on the unit sphere
    means = []
    for _ in range(n_groups):
        v = np.random.normal(size=x_dim)
        means.append(v / np.linalg.norm(v))
    means = np.vstack(means)  # shape (n_groups, x_dim)

    # 2) Build covariances
    covs = [make_ar1_cov(rho_list[i], x_dim, sigma_list[i]) for i in range(n_groups)]

    # 3) Sample group labels
    groups = np.random.choice(n_groups, size=size_sample, p=prop_list)

    # 4) Generate X by group
    X = np.zeros((size_sample, x_dim))
    for g in range(n_groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 0:
            X[idx] = np.random.multivariate_normal(means[g], covs[g], size=len(idx))

    dfX = pd.DataFrame(X, columns=[f"x{j}" for j in range(x_dim)])

    # 5) Set individual treatment effect
    tau = np.array([ite_list[g] for g in groups])

    # 6) Random treatment assignment
    if non_treated_frac is not None:
        n_control = int(non_treated_frac * size_sample)
        # pick lowest-risk (random here) to be control
        perm = np.random.permutation(size_sample)
        Z = np.ones(size_sample, dtype=int)
        Z[perm[:n_control]] = 0
    else:
        Z = np.random.binomial(1, 0.5, size_sample)

    # 7) Generate potential outcomes
    # Here baseline mu0 is zero
    Y0 = np.random.normal(0, noise_std, size_sample)
    Y1 = Y0 + tau

    # Observed outcome
    Y_cont = Y0 * (1 - Z) + Y1 * Z
    if binary:
        pY = 1 / (1 + np.exp(-Y_cont))
        Y = np.random.binomial(1, pY)
    else:
        Y = Y_cont

    # 8) Train-test split
    idx_all = np.arange(size_sample)
    np.random.shuffle(idx_all)
    n_train = int(train_ratio * size_sample)
    tr_idx, te_idx = idx_all[:n_train], idx_all[n_train:]

    def split(a): return a[tr_idx], a[te_idx]

    x_train, x_test = dfX.iloc[tr_idx], dfX.iloc[te_idx]
    z_train, z_test = split(Z)
    y_train, y_test = split(Y)
    group_train, group_test = split(groups)
    ite_test = tau[te_idx]
    catt = ite_test[z_test == 1]
    catc = ite_test[z_test == 0]

    return {
        "x_train": x_train,   "z_train": z_train,   "y_train": y_train, "group_train": group_train,
        "x_test": x_test,     "z_test": z_test,     "y_test": y_test, "group_test": group_test,
        "Test_ITE": ite_test, "Test_CATT": catt,    "Test_CATC": catc, "group_means" : means
    }


def analyze_mixture(
    data: dict,
    rho_list: list[float],
    sigma_list: list[float],
    save_prefix: str = "mixture_analysis",
    save_fig: bool = True
):
    """
    Analyze and visualize group separation from a mixture-data dict.

    Parameters
    ----------
    data : dict
      Output of prepare_train_data_mixture1 (must contain keys
      "x_test", "group_test", "group_means").
    rho_list : list of float
      The AR(1) rho parameter for each group in the same order as `group_means`.
    sigma_list : list of float
      The AR(1) marginal sigma for each group.
    save_prefix : str
      File-prefix for the saved plots (will write
      "{save_prefix}_2d.png" and "{save_prefix}_3d.png").
    """
    X_test     = data["x_test"].values
    groups     = data["group_test"]
    means      = data["group_means"]
    n_groups   = means.shape[0]
    d          = X_test.shape[1]

    # 1) Pairwise metrics
    DM   = np.zeros((n_groups, n_groups))
    DB   = np.zeros((n_groups, n_groups))
    rho_ov = np.zeros((n_groups, n_groups))
    cosS  = means @ means.T

    dets = []
    covs = []
    for i in range(n_groups):
        Ci = make_ar1_cov(rho_list[i], d, sigma_list[i])
        covs.append(Ci)
        dets.append(np.linalg.det(Ci))

    for i in range(n_groups):
        for j in range(i+1, n_groups):
            Sij    = 0.5 * (covs[i] + covs[j])
            invSij = np.linalg.inv(Sij)
            diff   = means[i] - means[j]

            DM[i,j]      = DM[j,i]      = np.sqrt(diff @ invSij @ diff)
            term1        = 0.125 * diff @ invSij @ diff
            term2        = 0.5   * np.log(np.linalg.det(Sij) / np.sqrt(dets[i]*dets[j]))
            DB[i,j]      = DB[j,i]      = term1 + term2
            rho_ov[i,j]  = rho_ov[j,i]  = np.exp(-DB[i,j])

    print("Cosine similarities:\n", np.round(cosS, 3))
    print("Mahalanobis distances:\n", np.round(DM, 3))
    print("Bhattacharyya distances:\n", np.round(DB, 3))
    print("Overlap coefficients:\n", np.round(rho_ov, 3))

    # 2) Silhouette on test set
    sil = silhouette_score(X_test, groups)
    print(f"\nSilhouette score (test): {sil:.3f}")

    # 3) Density‐based “recovery” error
    logdens = np.vstack([
        multivariate_normal.logpdf(X_test, mean=means[g], cov=covs[g])
        for g in range(n_groups)
    ]).T + np.log(1.0 / n_groups)
    pred = np.argmax(logdens, axis=1)
    err  = np.mean(pred != groups)
    conf = confusion_matrix(groups, pred)
    print(f"Misclassification rate: {err:.3f}")
    print("Confusion matrix:\n", conf)

    if save_fig:
        # 4) 2D random projection
        P2 = np.linalg.qr(np.random.randn(d, 2))[0]
        Y2 = X_test @ P2
        plt.figure(figsize=(6,6))
        plt.scatter(Y2[:,0], Y2[:,1], c=groups, s=1, alpha=0.5)
        plt.title("2D Random Projection")
        plt.xlabel("Comp 1"); plt.ylabel("Comp 2")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_2d.png", dpi=150)
        plt.close()

        # 5) 3D random projection
        P3 = np.linalg.qr(np.random.randn(d, 3))[0]
        Y3 = X_test @ P3
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(Y3[:,0], Y3[:,1], Y3[:,2], c=groups, s=1, alpha=0.5)
        ax.set_title("3D Random Projection")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_3d.png", dpi=150)
        plt.close()



if __name__ == "__main__":

    n_groups    = 3
    size_sample = 10000
    x_dim = 25
    sigma_list = [1.0, 1.0, 1.0]         # Marginal std dev for each of the 3 Gaussians
    rho_list   = [0.9, 0.9, 0.9]         # AR(1) correlation for each group
    ite_list   = [0.0, 2.0, -1.0]        # Constant ITE for each group
    prop_list  = [0.5, 0.3, 0.2]         # 50% in group0, 30% in group1, 20% in group2

    # 2. Generate the data
    output = prepare_train_data_mixture1(
        size_sample=size_sample,
        x_dim=x_dim,
        n_groups=n_groups,
        sigma_list=sigma_list,
        rho_list=rho_list,
        ite_list=ite_list,
        prop_list=prop_list,
        train_ratio=0.7,
        noise_std=1.0,
        binary=True,
        non_treated_frac=0.1
    )

    analyze_mixture(
         output,
         rho_list=rho_list,
         sigma_list=sigma_list,
         save_prefix="my_mixture"
     )

    def summarize_output(data_dict):
        print("=== x_train ===")
        print(data_dict["x_train"].describe())
        print("\n=== x_test ===")
        print(data_dict["x_test"].describe())

        print("\n=== Treatment (z_train) distribution ===")
        print(pd.Series(data_dict["z_train"]).value_counts(normalize=True))

        print("\n=== Outcome (y_train) summary ===")
        print(pd.Series(data_dict["y_train"]).describe())

        print("\n=== Test ITE Summary ===")
        print(pd.Series(data_dict["Test_ITE"]).describe())                                             

    summarize_output(output)              