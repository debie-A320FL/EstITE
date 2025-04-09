# Importing packages
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

from sklearn import preprocessing
from scipy import stats as sts
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
basedir = "/home/onyxia/work/EstITE/Simulations_Stage/Setup 1/Data"
data = pd.read_csv(basedir + "/simulated_1M_data.csv")

# The dataset is too large
# Randomly sample 1000 rows from the DataFrame
data = data.sample(n=1000, random_state=1)  # You can set a random_state for reproducibility
#AIDS = "/home/onyxia/work/EstITE/Simulations/ACTG/Data/ACTGData.csv"

# To save result
# Define the full path
results_dir = os.path.join(basedir, "..", "Results")
# Create the directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Define treatment assignment
myZ = np.array(data["treatment"])
# Define response
myY = np.array(data["Y"])
# Convert X to array
myX = np.array(data.drop(columns=["treatment"]))

# Scale numeric
# to_scale = ["age", "wtkg", "preanti"]
# AIDS[to_scale] = preprocessing.scale(AIDS[to_scale])

# Pred and obs
N, P = data.drop(columns=["treatment"]).shape

# Importer les hyperparamètres
hyperparams = pd.read_csv(basedir + "/hyperparams.csv")

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

# Ajouter une colonne pi pour la probabilité théorique
# data['pi'] = 1 / (1 + np.exp(-(mu_0 + tau * myZ)))

# Calculer ITE_proba
ITE_proba = 1 / (1 + np.exp(-(mu_0 + tau))) - 1 / (1 + np.exp(-mu_0))


# Results storage
# esti = ['CATT', 'CATC']; subs = ['Train', 'Test']; loss = ['Bias', 'PEHE', 'RLOSS']
esti = ['CATT', 'CATC']; subs = ['Train', 'Test']; loss = ['Bias', 'PEHE'] # RLOSS not stored atm

Results = {}
for i in range(2):
    for k in range(2):
        for j in loss:
            dest = {'%s_%s_%s' % (esti[i], subs[k], j): np.zeros((B, 2))}
            Results.update(dest)


##### Simulation Study
start = time.time()

for i in range(B):

    print("\n*** Iteration", i+1)

    # Set seed
    np.random.seed(100 + i)

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

    # 1) CMGP
    # print("CMGP")
    myCMGP = CMGP(dim=P, mode="CMGP", mod='Multitask', kern='RBF')
    #print("before fit")
    myCMGP.fit(X=x_train, Y=y_train, W=z_train)
    #print("after fit")

    train_CMGP_est = myCMGP.predict(x_train)[0]
    test_CMGP_est = myCMGP.predict(x_test)[0]

    # CATT
    Results['CATT_Train_Bias'][i, 0] = bias(CATT_Train, train_CMGP_est.reshape(-1)[z_train == 1])
    Results['CATT_Train_PEHE'][i, 0] = PEHE(CATT_Train, train_CMGP_est.reshape(-1)[z_train == 1])

    Results['CATT_Test_Bias'][i, 0] = bias(CATT_Test, test_CMGP_est.reshape(-1)[z_test == 1])
    Results['CATT_Test_PEHE'][i, 0] = PEHE(CATT_Test, test_CMGP_est.reshape(-1)[z_test == 1])

    # CATC
    Results['CATC_Train_Bias'][i, 0] = bias(CATC_Train, train_CMGP_est.reshape(-1)[z_train == 0])
    Results['CATC_Train_PEHE'][i, 0] = PEHE(CATC_Train, train_CMGP_est.reshape(-1)[z_train == 0])

    Results['CATC_Test_Bias'][i, 0] = bias(CATC_Test, test_CMGP_est.reshape(-1)[z_test == 0])
    Results['CATC_Test_PEHE'][i, 0] = PEHE(CATC_Test, test_CMGP_est.reshape(-1)[z_test == 0])
    
    
    # 2) NSGP
    myNSGP = CMGP(dim=P, mode="NSGP", mod='Multitask', kern='Matern')
    myNSGP.fit(X=x_train, Y=y_train, W=z_train)

    train_NSGP_est = myNSGP.predict(x_train)[0]
    test_NSGP_est = myNSGP.predict(x_test)[0]

    # CATT
    Results['CATT_Train_Bias'][i, 1] = bias(CATT_Train, train_NSGP_est.reshape(-1)[z_train == 1])
    Results['CATT_Train_PEHE'][i, 1] = PEHE(CATT_Train, train_NSGP_est.reshape(-1)[z_train == 1])

    Results['CATT_Test_Bias'][i, 1] = bias(CATT_Test, test_NSGP_est.reshape(-1)[z_test == 1])
    Results['CATT_Test_PEHE'][i, 1] = PEHE(CATT_Test, test_NSGP_est.reshape(-1)[z_test == 1])

    # CATC
    Results['CATC_Train_Bias'][i, 1] = bias(CATC_Train, train_NSGP_est.reshape(-1)[z_train == 0])
    Results['CATC_Train_PEHE'][i, 1] = PEHE(CATC_Train, train_NSGP_est.reshape(-1)[z_train == 0])

    Results['CATC_Test_Bias'][i, 1] = bias(CATC_Test, test_NSGP_est.reshape(-1)[z_test == 0])
    Results['CATC_Test_PEHE'][i, 1] = PEHE(CATC_Test, test_NSGP_est.reshape(-1)[z_test == 0])


elapsed = time.time() - start
print("\n\nElapsed time (in h) is", round(elapsed/3600, 2)) # 2h for B=100

models = ['CMGP', 'NSGP']
summary = {}

for name in Results.keys():
    PD_results = pd.DataFrame(Results[name], columns=models)
    PD_results.to_csv(basedir + "/../Results/GP_%s_%s.csv" % (B, name), index=False, header=True)

    aux = {name: {'CMGP': np.c_[np.mean(PD_results['CMGP']), MC_se(PD_results['CMGP'], B)],
                  'NSGP': np.c_[np.mean(PD_results['NSGP']), MC_se(PD_results['NSGP'], B)]}}
    summary.update(aux)

print(pd.DataFrame(summary).T)
print("\n\n++++++++  FINISHED  +++++++++")
