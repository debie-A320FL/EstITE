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
    return np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))))


def PEHE(T_true, T_est):
    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))) ** 2))


def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * np.std(np.array(x)) / np.sqrt(B)


def r_loss(y, mu, z, pi, tau):
    return np.mean( ( (y - mu) - (z - pi)*tau )**2 )


# Options
B = 100  # Num of simulations

# Load AIDS data
#basedir = str(Path(os.getcwd()).parents[2])
basedir = "/home/onyxia/work/EstITE/Simulations"
AIDS = pd.read_csv(basedir + "/ACTG/Data/ACTGData.csv")
#AIDS = "/home/onyxia/work/EstITE/Simulations/ACTG/Data/ACTGData.csv"

# To save result
# Define the full path
results_dir = os.path.join(basedir, "ACTG", "Results")
# Create the directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Define treatment assignment
myZ = np.array(AIDS["treat"])

# Scale numeric
to_scale = ["age", "wtkg", "preanti"]
AIDS[to_scale] = preprocessing.scale(AIDS[to_scale])

# Pred and obs
N, P = AIDS.drop(columns=["treat"]).shape

# Convert X to array
myX = np.array(AIDS.drop(columns=["treat"]))

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

    # Generate Surfaces
    mu = 8 - 0.5*AIDS["hemo"] - 1*np.abs(AIDS["wtkg"] - 1) + 0.5*AIDS["gender"] - 0.2/(AIDS["age"] + 2) + \
         0.5*AIDS["karnof_hi"] - 0.5*AIDS["z30"] - 0.5*AIDS["race"]

    ITE = 1 + 0.2*AIDS["wtkg"] + 2*sts.norm.pdf(AIDS["wtkg"])*(AIDS["karnof_hi"] + 2)

    sigma = (np.max(mu) - np.min(mu))/8
    myY = mu + ITE*myZ + sts.norm.rvs(0, sigma, N)

    # Train-Test Split (70-30%)
    split = np.random.choice(np.array([True, False]), N, replace=True, p=np.array([0.7, 0.3]))

    x_train = np.array(myX[split])
    x_test = np.array(myX[~split])

    y_train = np.array(myY[split])
    y_test = np.array(myY[~split])

    z_train = np.array(myZ[split])
    z_test = np.array(myZ[~split])

    ITE_train = np.array(ITE[split])
    ITE_test = np.array(ITE[~split])

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
    PD_results.to_csv(basedir + "/ACTG/Results/GP_%s_%s.csv" % (B, name), index=False, header=True)

    aux = {name: {'CMGP': np.c_[np.mean(PD_results['CMGP']), MC_se(PD_results['CMGP'], B)],
                  'NSGP': np.c_[np.mean(PD_results['NSGP']), MC_se(PD_results['NSGP'], B)]}}
    summary.update(aux)

print(pd.DataFrame(summary).T)
print("\n\n++++++++  FINISHED  +++++++++")
