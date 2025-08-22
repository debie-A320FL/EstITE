from scipy.stats import multivariate_normal
import numpy as np

def make_ar1_cov(rho: float, d: int, sigma: float = 1.0) -> np.ndarray:
    """
    Build a d×d AR(1) covariance matrix with parameter rho and marginal std sigma.
    """
    idx = np.arange(d)
    return sigma * (rho ** np.abs(idx[:, None] - idx[None, :]))

def estimate_bayes_risk(N: int, dim: int, sigma: float, n_groups: int = 3, 
                        rho: float = 0.5, seed: int = 123) -> float:
    """
    Estimate Bayes risk (misclassification rate) for a Gaussian mixture model.

    Parameters
    ----------
    N : int
        Number of data points to sample.
    dim : int
        Dimension of covariates.
    sigma : float
        Marginal std (coefficient on diagonal of covariance).
    n_groups : int, default=3
        Number of mixture components.
    rho : float, default=0.5
        AR(1) correlation parameter (same for all groups).
    seed : int, default=123
        Random seed.

    Returns
    -------
    float
        Estimated Bayes risk (misclassification probability).
    """
    np.random.seed(seed)

    # Equal mixture proportions
    prop_list = [1.0 / n_groups] * n_groups

    # Random means on the sphere
    means = []
    for _ in range(n_groups):
        v = np.random.normal(size=dim)
        means.append(v / np.linalg.norm(v))
    means = np.vstack(means)

    # Build covariance matrices
    covs = [make_ar1_cov(rho, dim, sigma) for _ in range(n_groups)]

    # Sample group labels
    groups = np.random.choice(n_groups, size=N, p=prop_list)

    # Generate data
    X = np.zeros((N, dim))
    for g in range(n_groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 0:
            X[idx] = np.random.multivariate_normal(means[g], covs[g], size=len(idx))

    # Bayes classification: pick class with max posterior density
    logpdfs = np.vstack([
        multivariate_normal.logpdf(X, mean=means[g], cov=covs[g])
        for g in range(n_groups)
    ]).T + np.log(1.0 / n_groups)
    pred = np.argmax(logpdfs, axis=1)

    # Misclassification rate
    risk = np.mean(pred != groups)
    return risk

# Repeat Bayes risk estimation 10 times
def repeat_estimation(N, dim, sigma, n_runs=10):
    risks = [estimate_bayes_risk(N, dim, sigma, seed=123 + i) for i in range(n_runs)]
    return np.mean(risks), np.std(risks)

dims = [5,10,25,50,100,300,500,1000]
mean_risks = []
std_risks  = []

for d in dims:
    print(f"dim = {d}")
    m, s = repeat_estimation(N=10000, dim=d, sigma=1.0, n_runs=10)
    mean_risks.append(m)
    std_risks.append(s)

for d, m, s in zip(dims, mean_risks, std_risks):
    print(f"dim={d:2d} -> Bayes risk: {m:.3f} ± {s:.3f}")

sigmas = [0.2, 0.3, 0.5, 1, 2, 3,5,7,10]
mean_risks = []
std_risks  = []

for sgm in sigmas:
    print(f"sigm = {sgm}")
    m, s = repeat_estimation(N=10000, dim=25, sigma=sgm, n_runs=10)
    mean_risks.append(m)
    std_risks.append(s)

for sgm, m, s in zip(sigmas, mean_risks, std_risks):
    print(f"sigma={sgm:.1f} -> Bayes risk: {m:.3f} ± {s:.3f}")