import os
import math
import random
import copy
from collections import deque
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from scipy.special import expit  # the logistic sigmoid

def train_group(
    X, y,
    hidden_dim=64,
    lr=0.001,
    max_iter=1000,
    tol=1e-3,
    patience=10,
    patience_lr=20,
    factor_lr=0.25,
    binary=False
):
    """
    Generic training + rollback patience, with optional GPU support.
    Returns: model, scaler, train_losses, val_losses, rollback_epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose loss fn
    if binary:
        loss_fn = nn.BCEWithLogitsLoss()
        print("Using binary loss")
    else:
        loss_fn = nn.MSELoss()
        print("Using continuous loss")

    

    # split + scale
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np.astype(np.float32))
    X_val_np = scaler.transform(X_val_np.astype(np.float32))

    # to tensors on device
    X_tr = torch.from_numpy(X_tr_np).to(device)
    y_tr = torch.from_numpy(y_tr_np.astype(np.float32)).to(device)
    X_val = torch.from_numpy(X_val_np).to(device)
    y_val = torch.from_numpy(y_val_np.astype(np.float32)).to(device)

    # model → device
    input_dim = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=factor_lr,
        patience=patience_lr, cooldown=2*patience_lr
    )

    prev_lr = optimizer.param_groups[0]['lr']
    train_losses, val_losses = [], []
    best_val, no_imp = float('inf'), 0
    recent_states = deque(maxlen=patience)
    rollback_epoch = None

    for epoch in range(1, max_iter + 1):
        # — training —
        model.train()
        optimizer.zero_grad()
        # loss_tr = nn.MSELoss()(model(X_tr), y_tr)
        logits = model(X_tr)            # raw outputs, shape (N,1)
        loss_tr = loss_fn(logits, y_tr)
        loss_tr.backward()
        optimizer.step()

        train_losses.append(loss_tr.item())
        recent_states.append(copy.deepcopy(model.state_dict()))

        # — validation —
        model.eval()
        with torch.no_grad():
            # loss_val = nn.MSELoss()(model(X_val), y_val).item()
            val_logits = model(X_val)
            loss_val = loss_fn(val_logits, y_val).item()
        val_losses.append(loss_val)

        # LR scheduler
        scheduler.step(loss_val)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr != prev_lr:
            print(f"[Epoch {epoch}] LR changed: {prev_lr:.3e} → {curr_lr:.3e}")
            prev_lr = curr_lr

        # early‐stop / rollback
        rel_imp = (best_val - loss_val)/best_val if best_val < float('inf') else float('inf')
        if rel_imp >= tol:
            best_val, no_imp = loss_val, 0
        else:
            no_imp += 1

        if no_imp >= patience:
            rollback_epoch = epoch - patience
            model.load_state_dict(recent_states[0])
            print(f"Stopping at epoch {epoch} — rolling back to epoch {rollback_epoch}.")
            break

        if epoch == max_iter:
            rollback_epoch = max_iter
            print(f"Reached max_iter={max_iter}, using epoch {rollback_epoch}.")

    return model, scaler, train_losses, val_losses, rollback_epoch


def train_multitask_group(
    X, Z, y,
    hidden_dim=64,
    lr=1e-3,
    max_iter=1000,
    tol=1e-3,
    patience=10,
    patience_lr=20,
    factor_lr=0.25,
    binary=False
):
    """
    Multi‐task training + rollback patience, with optional GPU support.
    If binary=True, uses BCEWithLogitsLoss on each head; else uses MSE.
    Returns: model, scaler, train_losses, val_losses, rollback_epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split + scale
    X_tr_np, X_val_np, Z_tr_np, Z_val_np, y_tr_np, y_val_np = train_test_split(
        X, Z, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np.astype(np.float32))
    X_val_np = scaler.transform(X_val_np.astype(np.float32))

    # to tensors
    X_tr = torch.from_numpy(X_tr_np).to(device)
    Z_tr = torch.from_numpy(Z_tr_np.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_tr_np.astype(np.float32)).to(device)
    X_val = torch.from_numpy(X_val_np).to(device)
    Z_val = torch.from_numpy(Z_val_np.astype(np.float32)).to(device)
    y_val = torch.from_numpy(y_val_np.astype(np.float32)).to(device)

    # ensure shape (N,1)
    if y_tr.dim() == 1:
        y_tr = y_tr.unsqueeze(1)
    if y_val.dim() == 1:
        y_val = y_val.unsqueeze(1)

    # model → device
    input_dim = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2),   # two heads: control & treated
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=factor_lr,
        patience=patience_lr, cooldown=2*patience_lr
    )

    prev_lr = optimizer.param_groups[0]['lr']
    train_losses, val_losses = [], []
    best_val, no_imp = float('inf'), 0
    recent_states = deque(maxlen=patience)
    rollback_epoch = None

    if binary:
        print("Using binary loss")
    else:
        print("Using continuous loss")

    for epoch in range(1, max_iter+1):
        # — TRAIN —
        model.train()
        optimizer.zero_grad()
        out_tr = model(X_tr)               # [N,2]
        target_tr = y_tr.repeat(1, 2)      # [N,2]
        mask_tr = torch.cat([1 - Z_tr, Z_tr], dim=1)  # select correct head

        if binary:
            # per-element BCE, then mask & average
            loss_mat = F.binary_cross_entropy_with_logits(
                out_tr, target_tr, reduction='none'
            )
            loss_tr = (loss_mat * mask_tr).sum(dim=1).mean()
        else:
            # masked MSE
            loss_tr = ((out_tr - target_tr)**2 * mask_tr).sum(dim=1).mean()

        loss_tr.backward()
        optimizer.step()

        train_losses.append(loss_tr.item())
        recent_states.append(copy.deepcopy(model.state_dict()))

        # — VALIDATION —
        model.eval()
        with torch.no_grad():
            out_val = model(X_val)
            target_val = y_val.repeat(1, 2)
            mask_val = torch.cat([1 - Z_val, Z_val], dim=1)

            if binary:
                loss_mat = F.binary_cross_entropy_with_logits(
                    out_val, target_val, reduction='none'
                )
                loss_val = (loss_mat * mask_val).sum(dim=1).mean().item()
            else:
                loss_val = ((out_val - target_val)**2 * mask_val).sum(dim=1).mean().item()

        val_losses.append(loss_val)

        # LR scheduler
        scheduler.step(loss_val)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr != prev_lr:
            print(f"[Epoch {epoch}] LR changed: {prev_lr:.3e} → {curr_lr:.3e}")
            prev_lr = curr_lr

        # EARLY STOP + ROLLBACK
        rel_imp = ((best_val - loss_val) / best_val) if best_val < float('inf') else float('inf')
        if rel_imp >= tol:
            best_val, no_imp = loss_val, 0
        else:
            no_imp += 1

        if no_imp >= patience:
            rollback_epoch = epoch - patience
            model.load_state_dict(recent_states[0])
            print(f"M-learner: stopping at epoch {epoch}, rolling back to {rollback_epoch}.")
            break

        if epoch == max_iter:
            rollback_epoch = max_iter
            print(f"M-learner: reached max_iter={max_iter}, using epoch {rollback_epoch}.")

    return model, scaler, train_losses, val_losses, rollback_epoch

def train_s_learner(X, Z, y, **train_kwargs):
    Xz = np.concatenate([X, Z], axis=1)
    model, scaler, tr, val, roll = train_group(Xz, y, **train_kwargs)
    print(f"S-learner: parameters from epoch {roll}\n")
    return model, scaler, tr, val


def train_t_learner(X, Z, y, **train_kwargs):
    X0, y0 = X[Z[:,0]==0], y[Z[:,0]==0]
    X1, y1 = X[Z[:,0]==1], y[Z[:,0]==1]
    m0, s0, t0_tr, t0_val, r0 = train_group(X0, y0, **train_kwargs)
    print(f"T-learner (Z=0): parameters from epoch {r0}")
    m1, s1, t1_tr, t1_val, r1 = train_group(X1, y1, **train_kwargs)
    print(f"T-learner (Z=1): parameters from epoch {r1}\n")
    return m0, s0, t0_tr, t0_val, m1, s1, t1_tr, t1_val


def train_x_learner(
    X,
    Z,
    y,
    compute_t: bool = True,
    t_models: tuple = None,
    **train_kwargs       # gathers hidden_dim, lr, max_iter, tol, patience, patience_lr, factor_lr, binary, etc.
):
    """
    X-learner with two τ-models, GPU-enabled if available.
    Any keyword in train_kwargs (e.g. binary=True, loss_type="bce",
    hidden_dim, lr, max_iter, tol, patience, ...) will be forwarded
    to train_t_learner() and train_group().

    Returns:
      - prop_model: sklearn logistic regression for propensity
      - (m0, s0), (m1, s1): T-learner outcome models + scalers
      - tau0_model, scaler_tau0, tr0_losses, val0_losses, rollback0
      - tau1_model, scaler_tau1, tr1_losses, val1_losses, rollback1
    """
    # 1) fit propensity model (on CPU)
    prop_model = LogisticRegression().fit(X, Z.ravel())

    # 2) fit or unpack T-learner outcome models
    if compute_t:
        m0, s0, _tr0, _val0, m1, s1, _tr1, _val1 = train_t_learner(
            X, Z, y,
            **train_kwargs
        )
    else:
        m0, s0, m1, s1 = t_models

    # 3) move outcome models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m0, m1 = m0.to(device), m1.to(device)

    # 4) compute μ₀(x), μ₁(x) on entire X
    X_np = X.astype(np.float32)
    with torch.no_grad():
        X_t0 = torch.from_numpy(s0.transform(X_np)).to(device)
        mu0 = m0(X_t0).cpu().numpy().ravel()
        X_t1 = torch.from_numpy(s1.transform(X_np)).to(device)
        mu1 = m1(X_t1).cpu().numpy().ravel()

    # 5) build pseudo‐outcomes D₀ and D₁
    mask0 = (Z.ravel() == 0)
    mask1 = (Z.ravel() == 1)

    D0 = (mu1[mask0] - y.ravel()[mask0]).reshape(-1, 1)
    X0 = X[mask0]

    D1 = (y.ravel()[mask1] - mu0[mask1]).reshape(-1, 1)
    X1 = X[mask1]

    # 6) fit τ₀ via train_group
    tau0_model, scaler_tau0, tr0, val0, rb0 = train_group(
        X0, D0,
        **train_kwargs
    )
    tau0_model = tau0_model.to(device)
    print(f"X-learner τ₀: parameters from epoch {rb0}")

    # 7) fit τ₁ via train_group
    tau1_model, scaler_tau1, tr1, val1, rb1 = train_group(
        X1, D1,
        **train_kwargs
    )
    tau1_model = tau1_model.to(device)
    print(f"X-learner τ₁: parameters from epoch {rb1}")

    return (
        prop_model,
        (m0, s0), (m1, s1),
        tau0_model, scaler_tau0, tr0, val0, rb0,
        tau1_model, scaler_tau1, tr1, val1, rb1
    )


def train_m_learner(X, Z, y, **train_kwargs):
    model, scaler, tr, val, r = train_multitask_group(X, Z, y, **train_kwargs)
    print(f"M-learner: parameters from epoch {r}\n")
    return model, scaler, tr, val


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    import time
    import math
    import torch
    import numpy as np
    import random
    from sklearn.model_selection import train_test_split

    # decide on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    Nrow = int(1e4)
    nb_sim = 13
    for non_tr_percentage in [10]:

        # storage for per-run metrics
        pehe_records = []
        time_records = []

        # reproducibility + simulations
        for sim in range(nb_sim):
            print(f"\n\nRun {sim}")
            SEED = 2025 + sim
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # --- generate data ---
            n_samples, n_features = Nrow, 1
            X = np.random.rand(n_samples, n_features).astype(np.float32)
            Z = np.random.binomial(1, non_tr_percentage/100, size=(n_samples,1)).astype(np.float32)
            # y0 = (2*X + X*X).sum(axis=1,keepdims=True)

            # y1 = (X + 3*X*X).sum(axis=1,keepdims=True)
            #y1 = (2*X +  1.5 *X*X - 0.5 * np.sqrt(X)).sum(axis=1,keepdims=True)

            # define your raw scores for control/treatment
            score0 = (0.5 + 10*X + 50 * X**2).sum(axis=1)          # shape (n,)
            score1 = (- 0.5 * X - 20 *X**2).sum(axis=1)

            # map them to probabilities via sigmoid
            p0 = expit(score0)                        # in (0,1)
            p1 = expit(score1)

            # now draw Bernoulli with those probabilities
            y0 = np.random.binomial(1, p0).reshape(-1,1).astype(np.float32)
            y1 = np.random.binomial(1, p1).reshape(-1,1).astype(np.float32)

            y  = (1-Z)*y0 + Z*y1

            X_train, X_test, Z_train, Z_test, y_train, y_test, y0_t, y0_te, y1_t, y1_te = \
                train_test_split(X, Z, y, y0, y1, test_size=0.3, random_state=SEED)

            params = dict(
                max_iter=50000, tol=1e-2,
                hidden_dim=64, lr=0.01,
                patience=100, patience_lr=25,
                factor_lr=0.5, 
                binary=False
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

            # prepare models on device & eval
            for mdl in (m_s, m0, m1, m_m, tau0_m, tau1_m):
                mdl.to(device).eval()

            # define estimators
            def estimate_S(x_np):
                x0 = np.concatenate([x_np, [[0.]]], axis=1).astype(np.float32)
                x1 = np.concatenate([x_np, [[1.]]], axis=1).astype(np.float32)
                with torch.no_grad():
                    t0 = m_s(torch.from_numpy(sc_s.transform(x0)).to(device)).item()
                    t1 = m_s(torch.from_numpy(sc_s.transform(x1)).to(device)).item()
                return t0, t1

            def estimate_T(x_np):
                x_np = x_np.astype(np.float32)
                with torch.no_grad():
                    t0 = m0(torch.from_numpy(sc0.transform(x_np)).to(device)).item()
                    t1 = m1(torch.from_numpy(sc1.transform(x_np)).to(device)).item()
                return t0, t1

            def estimate_M(x_np):
                x_np = x_np.astype(np.float32)
                with torch.no_grad():
                    out = m_m(torch.from_numpy(sc_m.transform(x_np)).to(device)).cpu().numpy().ravel()
                return out[0], out[1]

            def estimate_X(x_np):
                x_np = x_np.astype(np.float32).reshape(1, -1)
                p = prop_model.predict_proba(x_np)[:,1].item()
                with torch.no_grad():
                    tau0 = tau0_m(torch.from_numpy(sc_tau0.transform(x_np)).to(device)).item()
                    tau1 = tau1_m(torch.from_numpy(sc_tau1.transform(x_np)).to(device)).item()
                return 0.0, (1-p)*tau0 + p*tau1


            # todo: make the pehe compute faster (parallelizing, etc)
            # --- compute PEHE on test set ---
            pehe = {}
            for name, fn in [('S',estimate_S),('T',estimate_T),('M',estimate_M),('X',estimate_X)]:
                errs = []
                for i in range(len(X_test)):
                    x_i = X_test[i:i+1]
                    true_eff = (y1_te[i] - y0_te[i]).item()
                    y0_hat, y1_hat = fn(x_i)
                    errs.append(((y1_hat - y0_hat) - true_eff)**2)
                pehe[name] = math.sqrt(np.mean(errs))

            true_cate = (y1_te - y0_te).ravel()  # shape (n_test,)

            mean_cate = np.mean(true_cate)
            pehe['Mean-CATE'] = np.sqrt(np.mean((mean_cate - true_cate)**2))

            pehe['Zero-CATE'] = np.sqrt(np.mean((0.0 - true_cate)**2))

            print("PEHE results:")
            for name, val in pehe.items():
                print(f"  {name}-learner: {val:.6e}")

            # --- record metrics ---
            pehe_records.append({'sim': sim, **pehe})
            time_records.append({'sim': sim, 'S': time_s, 'T': time_t, 'M': time_m, 'X': time_x})

            # --- save learning curves ---
            if save_learning_curve and sim < 5:
                fname = f'learning_curves_{sim}_STXM_tr_per_{non_tr_percentage}_size_{Nrow}.pdf'
                with PdfPages(fname) as pdf:
                    for title, tr, val in [
                        ('S-learner', s_tr, s_val),
                        ('T-learner (Z=0)', t0_tr, t0_val),
                        ('T-learner (Z=1)', t1_tr, t1_val),
                        ('M-learner', m_tr, m_val),
                        ('X-learner (Tau 0)', x0_tr, x0_val),
                        ('X-learner (Tau 1)', x1_tr, x1_val),
                    ]:
                        plt.figure()
                        plt.plot(tr, label='Train Loss')
                        plt.plot(val, label='Val Loss')
                        plt.yscale('log')
                        plt.title(title + ' Learning Curve')
                        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
                        pdf.savefig(); plt.close()
                print(f"Saved learning curves to {fname}")

        # --- end of all sims: build dataframes and export ---
        df_pehe  = pd.DataFrame(pehe_records).set_index('sim')
        df_times = pd.DataFrame(time_records).set_index('sim')
        df_pehe.to_csv(f'pehe_{nb_sim}_tr_per_{non_tr_percentage}_size_{Nrow}.csv')
        df_times.to_csv(f'times_{nb_sim}_tr_per_{non_tr_percentage}_size_{Nrow}.csv')
        print("\nWrote pehe.csv and times.csv")

        # --- compute and print Q1, median, Q3 ---
        def print_quartiles(df, title):
            q1 = df.quantile(0.25)
            m  = df.quantile(0.50)
            q3 = df.quantile(0.75)
            print(f"\n{title} statistics (Q1, median, Q3):")
            for col in df.columns:
                print(f"  {col}: {q1[col]:8.3e}, {m[col]:8.3e}, {q3[col]:8.3e}")

        print_quartiles(df_pehe,  "PEHE")
        print_quartiles(df_times, "Time")
