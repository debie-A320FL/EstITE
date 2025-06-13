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

def train_group(
    X, y,
    hidden_dim=64,
    lr=0.001,
    max_iter=1000,
    tol=1e-3,
    patience=10,
    patience_lr=20,
    factor_lr=0.25,
):
    """
    Generic training + rollback patience, with optional GPU support.
    Returns: model, scaler, train_losses, val_losses, rollback_epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        loss_tr = nn.MSELoss()(model(X_tr), y_tr)
        loss_tr.backward()
        optimizer.step()

        train_losses.append(loss_tr.item())
        recent_states.append(copy.deepcopy(model.state_dict()))

        # — validation —
        model.eval()
        with torch.no_grad():
            loss_val = nn.MSELoss()(model(X_val), y_val).item()
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
):
    """
    Multi‐task training + rollback patience, with optional GPU support.
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

    # to tensors on device
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
        nn.Linear(hidden_dim, 2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=factor_lr,
        patience=patience_lr, cooldown=2*patience_lr
    )

    prev_lr = optimizer.param_groups[0]['lr']
    train_losses, val_losses = [], []
    best_val, no_imp   = float('inf'), 0
    recent_states      = deque(maxlen=patience)
    rollback_epoch     = None

    for epoch in range(1, max_iter+1):
        # — train —
        model.train()
        optimizer.zero_grad()
        out_tr = model(X_tr)                # [N,2]
        target_tr = y_tr.repeat(1, 2)       # [N,2]
        mask_tr   = torch.cat([1 - Z_tr, Z_tr], dim=1)
        loss_tr = ((out_tr - target_tr)**2 * mask_tr).sum(dim=1).mean()
        loss_tr.backward()
        optimizer.step()

        train_losses.append(loss_tr.item())
        recent_states.append(copy.deepcopy(model.state_dict()))

        # — val —
        model.eval()
        with torch.no_grad():
            out_val = model(X_val)
            target_val = y_val.repeat(1, 2)
            mask_val   = torch.cat([1 - Z_val, Z_val], dim=1)
            loss_val = ((out_val - target_val)**2 * mask_val).sum(dim=1).mean().item()

        val_losses.append(loss_val)
        scheduler.step(loss_val)

        # LR logging
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
    X, Z, y,
    compute_t: bool = True,
    t_models: tuple = None,
    hidden_dim=64,
    lr=1e-3,
    max_iter=1000,
    tol=1e-3,
    patience=10,
    patience_lr=20,
    factor_lr=0.25,
):
    """
    X-learner with two τ-models, GPU-enabled if available.

    Returns:
      prop_model,
      (m0, scaler0), (m1, scaler1),
      tau0_model, scaler_tau0, tr0, val0, rollback0,
      tau1_model, scaler_tau1, tr1, val1, rollback1
    """
    import torch
    from sklearn.linear_model import LogisticRegression

    # 0) pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) estimate propensity on CPU (scikit-learn)
    prop_model = LogisticRegression().fit(X, Z.ravel())

    # 2) (re)compute or unpack T-learner outcome models
    if compute_t:
        m0, s0, _, _, m1, s1, _, _ = train_t_learner(
            X, Z, y,
            hidden_dim=hidden_dim, lr=lr,
            max_iter=max_iter, tol=tol,
            patience=patience, patience_lr=patience_lr,
            factor_lr=factor_lr
        )
    else:
        m0, s0, m1, s1 = t_models

    # 3) move T-models to device
    m0 = m0.to(device)
    m1 = m1.to(device)

    # 4) compute mu0(x), mu1(x) on all X
    X_np = X.astype(np.float32)
    with torch.no_grad():
        X_t0 = torch.from_numpy(s0.transform(X_np)).to(device)
        mu0 = m0(X_t0).cpu().numpy().ravel()
        X_t1 = torch.from_numpy(s1.transform(X_np)).to(device)
        mu1 = m1(X_t1).cpu().numpy().ravel()

    # 5) build pseudo-outcomes with correct formulas
    mask0 = (Z.ravel() == 0)
    mask1 = (Z.ravel() == 1)

    # control group: D0 = μ1(x) – Y, treated group: D1 = Y – μ0(x)
    D0 = (mu1[mask0] - y.ravel()[mask0]).reshape(-1, 1)
    X0 = X[mask0]

    D1 = (y.ravel()[mask1] - mu0[mask1]).reshape(-1, 1)
    X1 = X[mask1]

    # 6) fit τ0 and τ1 via train_group (which itself will use GPU if available)
    tau0_model, scaler_tau0, tr0, val0, rb0 = train_group(
        X0, D0,
        hidden_dim=hidden_dim, lr=lr,
        max_iter=max_iter, tol=tol,
        patience=patience, patience_lr=patience_lr,
        factor_lr=factor_lr
    )
    tau0_model = tau0_model.to(device)
    print(f"X-learner τ0: parameters from epoch {rb0}")

    tau1_model, scaler_tau1, tr1, val1, rb1 = train_group(
        X1, D1,
        hidden_dim=hidden_dim, lr=lr,
        max_iter=max_iter, tol=tol,
        patience=patience, patience_lr=patience_lr,
        factor_lr=factor_lr
    )
    tau1_model = tau1_model.to(device)
    print(f"X-learner τ1: parameters from epoch {rb1}")

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

    Nrow = int(1e5)
    nb_sim = 80
    for non_tr_percentage in [25, 10, 5, 2]:

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
            y0 = (2*X + X*X).sum(axis=1,keepdims=True)

            y1 = (X + 3*X*X).sum(axis=1,keepdims=True)
            #y1 = (2*X +  1.5 *X*X - 0.5 * np.sqrt(X)).sum(axis=1,keepdims=True)
            y  = (1-Z)*y0 + Z*y1

            X_train, X_test, Z_train, Z_test, y_train, y_test, y0_t, y0_te, y1_t, y1_te = \
                train_test_split(X, Z, y, y0, y1, test_size=0.3, random_state=SEED)

            params = dict(
                max_iter=15000, tol=1e-3,
                hidden_dim=64, lr=0.01,
                patience=100, patience_lr=25,
                factor_lr=0.5
            )

            save_learning_curve = False
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

            print("PEHE results:")
            for name, val in pehe.items():
                print(f"  {name}-learner: {val:.6e}")

            # --- record metrics ---
            pehe_records.append({'sim': sim, **pehe})
            time_records.append({'sim': sim, 'S': time_s, 'T': time_t, 'M': time_m, 'X': time_x})

            # --- save learning curves ---
            if save_learning_curve:
                fname = f'learning_curves_{sim}_STXM.pdf'
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
