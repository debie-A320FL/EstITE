# Importing packages
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

import statsmodels.api as sm
import patsy



from sklearn import preprocessing
from scipy import stats as sts

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Ajouter le chemin du dossier Python de Setup 1 à sys.path
import sys
setup_1_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Setup 1a/Code/Python'))
# print(setup_1_models_path)
sys.path.append(setup_1_models_path)

from nn_learner_draft import *

from prepare_data import *

# Evaluation Functions
def bias(T_true, T_est):
    return np.mean(100*T_true.reshape((-1, 1)) - 100*T_est.reshape((-1, 1)))


def PEHE(T_true, T_est):
    return np.sqrt(np.mean((100*T_true.reshape((-1, 1)) - 100*T_est.reshape((-1, 1))) ** 2))


def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * np.std(np.array(x)) / np.sqrt(B)


def r_loss(y, mu, z, pi, tau):
    return np.mean( ( (y - mu) - (z - pi)*tau )**2 )


# Load AIDS data
#basedir = str(Path(os.getcwd()).parents[2])
# Utilisation des données de setup 1
basedir_setup_1 = "/home/onyxia/work/EstITE/Simulations_Stage/Setup 1a/Data"

os.chdir("/home/onyxia/work/EstITE/Simulations_Stage/Setup 7")

# Load data
hyperparams_df = pd.read_csv("./../Setup 1a/Data/hyperparams.csv")
data_train_test = pd.read_csv("./../Setup 1a/Data/simulated_1M_data.csv")

# Convert hyperparams to dictionary (if it's a single row)
hyperparams = hyperparams_df.iloc[0].to_dict()

# treatment_percentile = 10
for x_dim in [25]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Options
    B = 20  # Num of simulations

    size_sample_list = [int(1e5)]
    size_sample = size_sample_list[0]

    sigma_single_list = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 5]

    for sigma in sigma_single_list:
        print(f"sigma = {sigma}")

        pehe_records = []
        time_records = []
        
        for sim in range(B):

            print(f"\n*** Iteration {sim + 1} - Size sample : {size_sample} - x_dim : {x_dim}")

            # Set seed
            np.random.seed(100 + sim)
            seed = (100 + sim)

            size_sample_train_test = size_sample

            #res_train_test = prepare_train_data(
            #    hyperparams=hyperparams,
            #    size_sample=size_sample_train_test,
            #    train_ratio=0.7,
            #    seed=seed,
            #    treatment_percentile=treatment_percentile,
            #    binary=binary,
            #    verbose=True
            #)

            #res_train_test = prepare_train_data_null_cate_indep_treatment(size_sample = size_sample, x_dim = x_dim, seed=sim,
            #                                     train_ratio=0.7, treatment_prob=0.1,
            #     
            # 
            #                                 binary=True)
            scen = 4
            if scen == 1:
                res_train_test = prepare_train_data_scenario1(
                            size_sample = size_sample,
                            x_dim = x_dim,
                            k_mu = x_dim //3,           # Number of features used in outcome model μ₀
                            k_conf = x_dim //3,         # Number of additional features used in π(x) only (confounders)
                            seed = sim,
                            train_ratio = 0.7,
                            noise_std  = 1.0,
                            binary = True
                        )
            elif scen == 2:
                res_train_test = prepare_train_data_scenario2(
                        size_sample = size_sample,
                        x_dim = x_dim,
                        k_mu = x_dim //4,            # Number of features used in outcome model μ₀
                        k_conf = x_dim //4,          # Additional features used in π(x)
                        k_tau = x_dim //4,           # Features used in τ(x)
                        seed = sim,
                        train_ratio = 0.7,
                        noise_std = 1.0,
                        binary = True,
                        non_treated_frac = 0.1  # e.g., 0.1 for forcing ~10% untreated
                    )
            elif scen == 3:
                res_train_test = prepare_train_data_scenario3(
                                size_sample = size_sample,
                                x_dim = x_dim,
                                k_mu = x_dim //4,             # Number of features for μ₀(x)
                                k_tau = x_dim //4,            # Number of features for μ₁(x)
                                k_pi = x_dim //4,             # Features affecting π(x)
                                seed = sim,
                                train_ratio = 0.7,
                                noise_std = 1.0,
                                binary = False,
                                non_treated_frac = 0.1  # e.g., 0.1 for forcing 10% untreated
                            )
            elif scen == 4:
                # 1. Specify your parameters
                n_groups    = 3

                sigma_list = [sigma, sigma , sigma]         # Marginal std dev for each of the 3 Gaussians
                rho_list   = [0, 0, 0]         # AR(1) correlation for each group
                ite_list   = [0.0, 2.0, -1.0]        # Constant ITE for each group
                prop_list  = [0.5, 0.3, 0.2]         # 50% in group0, 30% in group1, 20% in group2

                # 2. Generate the data
                res_train_test = prepare_train_data_mixture1(
                    size_sample=size_sample,
                    x_dim=x_dim,
                    n_groups=n_groups,
                    sigma_list=sigma_list,
                    rho_list=rho_list,
                    ite_list=ite_list,
                    prop_list=prop_list,
                    seed=seed,
                    train_ratio=0.7,
                    noise_std=1.0,
                    binary=True,
                    non_treated_frac=0.1
                )

                print("\nBeginning of the analysis")
                save_prefix = f"Figures/mixture_plot_{sim}_sigma_{sigma}"
                analyze_mixture(
                    res_train_test,
                    rho_list=rho_list,
                    sigma_list=sigma_list,
                    save_prefix=save_prefix,
                    save_fig=sim < 3
                )
                print("End of the analysis\n\n\n")

            X_train = res_train_test["x_train"]
            # print(X_train.head())
            Z_train = res_train_test["z_train"]
            y_train = res_train_test["y_train"]

            n0 = np.sum(Z_train == 0)
            n1 = np.sum(Z_train == 1)
            print(f"Count of Z=0: {n0}")
            print(f"Count of Z=1: {n1}")

            X_test = res_train_test["x_test"]
            Z_test = res_train_test["z_test"]
            y_test = res_train_test["y_test"]
            Test_CATT = res_train_test["Test_CATT"]
            Test_CATC = res_train_test["Test_CATC"]
            Test_ITE = res_train_test["Test_ITE"]

            # ensure flat NumPy arrays in the right shape & dtype
            X_train = np.asarray(X_train, dtype=np.float32)
            Z_train = np.asarray(Z_train, dtype=np.float32).reshape(-1, 1)
            y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

            #print(Z_train)

            X_test = np.asarray(X_test, dtype=np.float32)
            Z_test = np.asarray(Z_test, dtype=np.float32).reshape(-1, 1)
            y_test = np.asarray(y_test, dtype=np.float32).reshape(-1, 1)

            # NumPy arrays
            #print("X_train:", X_train.shape)   # e.g. (7000, 10)
            #print("Z_train:", Z_train.shape)   # e.g. (7000, 1)
            #print("y_train:", y_train.shape)   # e.g. (7000, 1)
            hidden_dim = 64
            params = dict(
                    max_iter=50000, tol=1e-2,
                    hidden_dim=hidden_dim, lr=0.01,
                    patience=100, patience_lr=25,
                    factor_lr=0.5
                )

            save_learning_curve = True
            print("Training S, T, M, X learners...\n")

            # --- S learner ---
            t0 = time.time()
            m_s, sc_s, s_tr, s_val = train_s_learner(X_train, Z_train, y_train, **params)
            t1 = time.time()
            time_s = t1 - t0
            print(f"S-learner predict_time : {round(time_s,3)}")

            # --- T learner ---
            t0 = time.time()
            m0, sc0, t0_tr, t0_val, m1, sc1, t1_tr, t1_val = train_t_learner(X_train, Z_train, y_train, **params)
            t1 = time.time()
            time_t = t1 - t0
            print(f"T-learner predict_time : {round(time_t,3)}")

            # --- M learner ---
            t0 = time.time()
            m_m, sc_m, m_tr, m_val = train_m_learner(X_train, Z_train, y_train, **params)
            t1 = time.time()
            time_m = t1 - t0
            print(f"M-learner predict_time : {round(time_m,3)}")

            # --- X learner ---
            t0 = time.time()
            (prop_model,
            (m0, sc0), (m1, sc1),
            tau0_m, sc_tau0, x0_tr, x0_val, rb0,
            tau1_m, sc_tau1, x1_tr, x1_val, rb1
            ) = train_x_learner(
                X_train, Z_train, y_train,
                compute_t=False, t_models=(m0, sc0, m1, sc1),
                **params
            )
            t1 = time.time()
            time_x = t1 - t0
            print(f"X-learner predict_time (without T part): {round(time_x,3)}")

            # --- RA learner ---
            t0 = time.time()
            tau_ra, sc_ra, ra_tr, ra_val, rb_ra = train_ra_learner(X_train, Z_train, y_train, **params)
            time_ra = time.time() - t0
            print(f"RA-learner predict_time : {time_ra:.3f}")

            # --- DR learner ---
            t0 = time.time()
            tau_dr, sc_dr, dr_tr, dr_val, rb_dr = train_dr_learner(X_train, Z_train, y_train, **params)
            time_dr = time.time() - t0
            print(f"DR-learner predict_time : {time_dr:.3f}")

            # --- R learner ---
            t0 = time.time()
            tau_r, sc_r, r_tr, r_val, rb_r = train_r_learner(X_train, Z_train, y_train, **params)
            time_r = time.time() - t0
            print(f"R-learner predict_time : {time_r:.3f}\n")

            # prepare models on device & eval
            for mdl in (m_s, m0, m1, m_m, tau0_m, tau1_m):
                mdl.to(device).eval()

            


            
            # --- compute PEHE for all learners (vectorized) ---
            true_eff = Test_ITE

            # 1) S-learner
            X0_s = np.hstack([X_test, np.zeros((len(X_test),1),dtype=np.float32)])
            X1_s = np.hstack([X_test, np.ones ((len(X_test),1),dtype=np.float32)])
            with torch.no_grad():
                t0_s = m_s(torch.from_numpy(sc_s.transform(X0_s)).to(device)).cpu().numpy().ravel()
                t1_s = m_s(torch.from_numpy(sc_s.transform(X1_s)).to(device)).cpu().numpy().ravel()
            pred_S = t1_s - t0_s

            # 2) T-learner
            with torch.no_grad():
                t0_t = m0(torch.from_numpy(sc0.transform(X_test.astype(np.float32))).to(device)).cpu().numpy().ravel()
                t1_t = m1(torch.from_numpy(sc1.transform(X_test.astype(np.float32))).to(device)).cpu().numpy().ravel()
            pred_T = t1_t - t0_t

            # 3) M-learner
            with torch.no_grad():
                out_m = m_m(torch.from_numpy(sc_m.transform(X_test.astype(np.float32))).to(device))
                t0_m_, t1_m_ = out_m[:,0].cpu().numpy(), out_m[:,1].cpu().numpy()
            pred_M = t1_m_ - t0_m_

            # 4) X-learner
            p = prop_model.predict_proba(X_test.astype(np.float32))[:,1]
            with torch.no_grad():
                tau0 = tau0_m(torch.from_numpy(sc_tau0.transform(X_test.astype(np.float32))).to(device)).cpu().numpy().ravel()
                tau1 = tau1_m(torch.from_numpy(sc_tau1.transform(X_test.astype(np.float32))).to(device)).cpu().numpy().ravel()
            pred_X = (1-p)*tau0 + p*tau1

            # 5) RA, DR, R learners
            with torch.no_grad():
                pred_RA = tau_ra(torch.from_numpy(sc_ra.transform(X_test.astype(np.float32))).to(device)).cpu().numpy().ravel()
                pred_DR = tau_dr(torch.from_numpy(sc_dr.transform(X_test.astype(np.float32))).to(device)).cpu().numpy().ravel()
                pred_R  = tau_r(torch.from_numpy(sc_r .transform(X_test.astype(np.float32))).to(device)).cpu().numpy().ravel()

            # helper
            def compute_pehe(pred):
                return math.sqrt(np.mean((pred - true_eff)**2))

            pehe = {
                'S-NN':  compute_pehe(pred_S),
                'T-NN':  compute_pehe(pred_T),
                'M-NN':  compute_pehe(pred_M),
                'X-NN':  compute_pehe(pred_X),
                'RA-NN': compute_pehe(pred_RA),
                'DR-NN': compute_pehe(pred_DR),
                'R-NN':  compute_pehe(pred_R),
                'ATE-learner': compute_pehe(np.full_like(true_eff, true_eff.mean())),
                'Zero-learner': compute_pehe(np.zeros_like(true_eff)),
            }
            
            print("PEHE results:")
            for name, val in pehe.items():
                print(f"  {name}: {val:.3e}")


            # --- record metrics ---
            pehe_records.append({'sim': sim, **pehe})
            time_records.append({
                'sim': sim,
                'S-NN': time_s, 'T-NN': time_t, 'M-NN': time_m, 'X-NN': time_x,
                'RA-NN': time_ra, 'DR-NN': time_dr, 'R-NN': time_r
            })
            # --- save learning curves ---
            if save_learning_curve and sim < 3:
                fname = f'learning_curves/learning_curves_{sim}_STXM_x_dim_{x_dim}_size_{size_sample}_scenario_{scen}_sigma_{sigma}.pdf'
                with PdfPages(fname) as pdf:
                    for title, tr, val in [
                        ('S-learner', s_tr, s_val),
                        ('T-learner (Z=0)', t0_tr, t0_val),
                        ('T-learner (Z=1)', t1_tr, t1_val),
                        ('M-learner', m_tr, m_val),
                        ('X-learner (Tau 0)', x0_tr, x0_val),
                        ('X-learner (Tau 1)', x1_tr, x1_val),
                        ('RA-learner', ra_tr, ra_val),
                        ('DR-learner', dr_tr, dr_val),
                        ('R-learner', r_tr, r_val),
                    ]:
                        plt.figure()
                        plt.plot(tr, label='Train Loss')
                        plt.plot(val, label='Val Loss')
                        plt.yscale('log')
                        plt.title(title + ' Learning Curve')
                        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
                        pdf.savefig(); plt.close()
                print(f"Saved learning curves to {fname}")


        # to complete

        # export perf model as csv
        # export time model as csv

        # print model performance : Q1, Med and Q3
        # print model time : Q1, Med and Q3

        results_dir = "./Results"
        # Create DataFrames
        # Build DataFrames
        df_pehe = pd.DataFrame(pehe_records)
        df_time = pd.DataFrame(time_records)

        # Save raw per-simulation results
        df_pehe.to_csv(os.path.join(results_dir, f"x_dim_{x_dim}_N_{size_sample}_scenario_{scen}_sigma_{sigma}.csv"), index=False)
        df_time.to_csv(os.path.join(results_dir, f"x_dim_{x_dim}_N_{size_sample}_scenario_{scen}_sigma_{sigma}.csv"), index=False)

        # Compute Q1, median, Q3 summaries
        summary_pehe = df_pehe.drop(columns='sim').quantile([0.25, 0.5, 0.75]).T
        summary_pehe.columns = ['Q1', 'Median', 'Q3']
        summary_time = df_time.drop(columns='sim').quantile([0.25, 0.5, 0.75]).T
        summary_time.columns = ['Q1', 'Median', 'Q3']

        # Combine and save summary
        summary = pd.concat({'PEHE': summary_pehe, 'Time(s)': summary_time}, axis=1)
        summary.to_csv(os.path.join(results_dir, f"summary_x_dim_{x_dim}_N_{size_sample}_scenario_{scen}_sigma_{sigma}.csv"))

        print("\nCombined Performance and Time Summary (rounded):")
        print(summary.round(4))