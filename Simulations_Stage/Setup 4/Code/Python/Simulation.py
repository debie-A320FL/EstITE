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

# Ajouter le chemin du dossier Python de Setup 1 à sys.path
import sys
setup_1_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Setup 1a/Code/Python'))
# print(setup_1_models_path)
sys.path.append(setup_1_models_path)

from models.causal_models import CMGP


# Evaluation Functions
def bias(T_true, T_est):
    return np.mean(100*T_true.reshape((-1, 1)) - 100*T_est.reshape((-1, 1)))


def PEHE(T_true, T_est):
    return np.sqrt(np.mean((100*T_true.reshape((-1, 1)) - 100*T_est.reshape((-1, 1))) ** 2))


def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * np.std(np.array(x)) / np.sqrt(B)


def r_loss(y, mu, z, pi, tau):
    return np.mean( ( (y - mu) - (z - pi)*tau )**2 )


# Options
B = 80  # Num of simulations

# Load AIDS data
#basedir = str(Path(os.getcwd()).parents[2])
# Utilisation des données de setup 1
basedir_setup_1 = "/home/onyxia/work/EstITE/Simulations_Stage/Setup 1a/Data"

#N_size = [1000, 2000, 3000, 5000, 10000]
N_size = [1e4, 5e4, 1e5, 5e5, 1e6]

for N in N_size:
    print(f"N = {N}")
    data = pd.read_csv(basedir_setup_1 + "/simulated_1M_data.csv")

    # The dataset is too large
    # Randomly sample 1000 rows from the DataFrame
    size_sample = int(N)
    data = data.sample(n=size_sample, random_state=1)  # You can set a random_state for reproducibility
    #AIDS = "/home/onyxia/work/EstITE/Simulations/ACTG/Data/ACTGData.csv"

    # To save result
    # Define the full path
    results_dir = "/home/onyxia/work/EstITE/Simulations_Stage/Setup 4/Results"
    # Create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Define treatment assignment
    myZ = np.array(data["treatment"])
    # Define response
    # myY = np.array(data["Y"])
    # Convert X to array
    myX = np.array(data.drop(columns=["treatment", "Y"]))

    # Scale numeric
    # to_scale = ["age", "wtkg", "preanti"]
    # AIDS[to_scale] = preprocessing.scale(AIDS[to_scale])

    # Pred and obs
    N, P = data.drop(columns=["treatment","Y"]).shape

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

    #print(f"tau (min,max) : {(min(tau), max(tau))}")

    # Générer le vecteur de bruit gaussien
    # bruit_gaussien = np.random.normal(0, hyperparams['sigma_sq'], size_sample)

    # Fonction logistique (logit inverse)
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    # data["Y_proba"] = logistic(ITE + bruit_gaussien)

    # myY = np.array(data["Y_proba"])

    # Convertir en DataFrame pandas
    # df = pd.DataFrame(data, columns=["Y_proba"])

    # Utiliser describe pour obtenir les statistiques
    # summary = df.describe()

    # Afficher le résumé
    # print(summary)



    # Ajouter une colonne pi pour la probabilité théorique
    # data['pi'] = 1 / (1 + np.exp(-(mu_0 + tau * myZ)))

    # Calculer ITE_proba
    ITE_proba = 1 / (1 + np.exp(-(mu_0 + tau))) - 1 / (1 + np.exp(-mu_0)) #todo utiliser logistic ?


    # Results storage
    # esti = ['CATT', 'CATC']; subs = ['Train', 'Test']; loss = ['Bias', 'PEHE', 'RLOSS']
    esti = ['CATT']; subs = ['Test']; loss = ['Bias', 'PEHE'] # RLOSS not stored atm

    Results = {}
    for i in range(1):
        for k in range(1):
            for j in loss:
                dest = {'%s_%s_%s' % (esti[i], subs[k], j): np.zeros((B, 3))}
                Results.update(dest)

    Time = np.zeros((B, 3))

    ##### Simulation Study
    # start = time.time()


    for i in range(B):

        print(f"\n*** Iteration {i+1} - Size sample : {N}")

        # Set seed
        np.random.seed(100 + i)

        # Générer le vecteur de bruit gaussien
        bruit_gaussien = np.random.normal(0, np.sqrt(hyperparams['sigma_sq']), size_sample)
        #bruit_gaussien = 0
        fac = 1
        data["Y_proba"] = logistic(ITE + bruit_gaussien * fac)

        myY = np.array(data["Y_proba"])

        # Train-Test Split (70-30%)
        split = np.random.choice(np.array([True, False]), N, replace=True, p=np.array([0.7, 0.3]))

        x_train = np.array(myX[split])
        x_test = np.array(myX[~split])

        y_train = np.array(myY[split])
        y_test = np.array(myY[~split])

        z_train = np.array(myZ[split])
        z_test = np.array(myZ[~split])

        ITE_train = np.array(ITE_proba[split])
        ITE_test = np.array(ITE_proba[~split])

        CATT_Train = ITE_train[z_train == 1]; CATC_Train = ITE_train[z_train == 0]
        CATT_Test = ITE_test[z_test == 1]; CATC_Test = ITE_test[z_test == 0]

        """
        # 1) CMGP
        
        start_time = time.time()
        myCMGP = CMGP(dim=P, mode="CMGP", mod='Multitask', kern='RBF')
        myCMGP.fit(X=x_train, Y=y_train, W=z_train)

        train_CMGP_est = myCMGP.predict(x_train)[0]
        test_CMGP_est = myCMGP.predict(x_test)[0]

        end_time = time.time()
        execution_time = end_time - start_time
        Time[i,0] = execution_time
        print(f"\nCMGP_predict_time : {round(execution_time,3)}")

        # CATT

        Results['CATT_Test_Bias'][i, 0] = bias(CATT_Test, test_CMGP_est.reshape(-1)[z_test == 1])
        Results['CATT_Test_PEHE'][i, 0] = PEHE(CATT_Test, test_CMGP_est.reshape(-1)[z_test == 1])

        print('CATT_Test_PEHE_CMGP')
        print(Results['CATT_Test_PEHE'][i, 0])


        # 2) NSGP
        start_time = time.time()
        myNSGP = CMGP(dim=P, mode="NSGP", mod='Multitask', kern='Matern')
        myNSGP.fit(X=x_train, Y=y_train, W=z_train)

        train_NSGP_est = myNSGP.predict(x_train)[0]
        test_NSGP_est = myNSGP.predict(x_test)[0]

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nNSGP_predict_time : {round(execution_time,3)}")
        Time[i,1] = execution_time

        # CATT
        Results['CATT_Test_Bias'][i, 1] = bias(CATT_Test, test_NSGP_est.reshape(-1)[z_test == 1])
        Results['CATT_Test_PEHE'][i, 1] = PEHE(CATT_Test, test_NSGP_est.reshape(-1)[z_test == 1])

        print('CATT_Test_PEHE_NSGP')
        print(Results['CATT_Test_PEHE'][i, 1])
        """

        # 3) Logistic Regression with Interaction Terms using patsy
        # Create DataFrames for train and test sets

        start_time = time.time()

        df_train = pd.DataFrame(x_train, columns=column_names)
        df_train['treatment'] = z_train
        df_train['y'] = y_train

        df_test = pd.DataFrame(x_test, columns=column_names)
        df_test['treatment'] = z_test
        df_test['y'] = y_test

        # Define the formula with interaction terms
        formula = 'y ~ age + weight + comorbidities + gender + treatment + treatment:age + treatment:weight + treatment:comorbidities + treatment:gender'

        # Create design matrices
        y_train_patsy, X_train_patsy = patsy.dmatrices(formula, df_train, return_type='dataframe')
        y_test_patsy, X_test_patsy = patsy.dmatrices(formula, df_test, return_type='dataframe')

        # start_time = time.time()
        # Fit logistic regression model
        logit_model = sm.Logit(y_train_patsy, X_train_patsy)
        logit_result = logit_model.fit(disp=0)

        # Create copies of the DataFrames with treatment set to 1 and 0
        df_train_t1 = df_train.copy()
        df_train_t0 = df_train.copy()
        df_test_t1 = df_test.copy()
        df_test_t0 = df_test.copy()

        df_train_t1['treatment'] = 1
        df_train_t0['treatment'] = 0
        df_test_t1['treatment'] = 1
        df_test_t0['treatment'] = 0

        # Recreate design matrices with updated treatment values
        _, X_train_patsy_t1 = patsy.dmatrices(formula, df_train_t1, return_type='dataframe')
        _, X_train_patsy_t0 = patsy.dmatrices(formula, df_train_t0, return_type='dataframe')
        _, X_test_patsy_t1 = patsy.dmatrices(formula, df_test_t1, return_type='dataframe')
        _, X_test_patsy_t0 = patsy.dmatrices(formula, df_test_t0, return_type='dataframe')

        # Predict probabilities for treatment and control
        train_logit_est_t1 = logit_result.predict(X_train_patsy_t1).values
        train_logit_est_t0 = logit_result.predict(X_train_patsy_t0).values
        test_logit_est_t1 = logit_result.predict(X_test_patsy_t1).values
        test_logit_est_t0 = logit_result.predict(X_test_patsy_t0).values

        # Compute treatment effect
        train_logit_est = train_logit_est_t1 - train_logit_est_t0
        test_logit_est = test_logit_est_t1 - test_logit_est_t0

        end_time = time.time()
        execution_time = end_time - start_time
        Time[i,2] = execution_time
        print(f"\nlogit_predict_time : {round(execution_time,3)}")

        # CATT
        Results['CATT_Test_Bias'][i, 2] = bias(CATT_Test, test_logit_est[z_test == 1])
        Results['CATT_Test_PEHE'][i, 2] = PEHE(CATT_Test, test_logit_est[z_test == 1])

        print('CATT_Test_PEHE_logit')
        print(Results['CATT_Test_PEHE'][i, 2])


    models = ['CMGP', 'NSGP', 'Logistic']
    summary = {}

    # Convert to DataFrame
    df_time = pd.DataFrame(Time, columns=models)
    # print(df)


    # Export to CSV
    
    df_time.to_csv(os.path.join(results_dir,f'Time_Nsize_{N}_B_{B}.csv'), index=False)
    
    
    print("Mean execution Time:")
    print(df_time.mean())

    for name in Results.keys():
        PD_results = pd.DataFrame(Results[name], columns=models)
        
        PD_results.to_csv(os.path.join(results_dir, "GP_%s_%s_Nsize_%s_fac_%s.csv" % (B, name, N, fac)), index=False, header=True)

        aux = {name: {'CMGP': np.c_[round(np.mean(PD_results['CMGP']),3), round(MC_se(PD_results['CMGP'], B),3)],
                    'NSGP': np.c_[round(np.mean(PD_results['NSGP']),3), round(MC_se(PD_results['NSGP'], B),3)],
                    'Logistic': np.c_[round(np.mean(PD_results['Logistic']),3), round(MC_se(PD_results['Logistic'], B),3)]}}
        summary.update(aux)

    print(pd.DataFrame(summary).T)
    print("\n\n++++++++  FINISHED  +++++++++")
