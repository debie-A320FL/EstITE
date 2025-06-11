import os
import math
import random
import copy
from collections import deque

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
    Generic training + rollback patience.
    Returns: model, scaler, train_losses, val_losses, rollback_epoch
    """
    # split
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # scale
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np.astype(np.float32))
    X_val_np = scaler.transform(X_val_np.astype(np.float32))
    # to tensors
    X_tr = torch.from_numpy(X_tr_np)
    y_tr = torch.from_numpy(y_tr_np.astype(np.float32))
    X_val = torch.from_numpy(X_val_np)
    y_val = torch.from_numpy(y_val_np.astype(np.float32))
    # model
    input_dim = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
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
        # train
        model.train()
        optimizer.zero_grad()
        loss_tr = nn.MSELoss()(model(X_tr), y_tr)
        loss_tr.backward()
        optimizer.step()
        train_losses.append(loss_tr.item())
        recent_states.append(copy.deepcopy(model.state_dict()))
        # val
        model.eval()
        with torch.no_grad():
            loss_val = nn.MSELoss()(model(X_val), y_val).item()
        val_losses.append(loss_val)
        # lr scheduler
        scheduler.step(loss_val)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr != prev_lr:
            print(f"[Epoch {epoch}] LR changed: {prev_lr:.3e} → {curr_lr:.3e}")
            prev_lr = curr_lr
        # improvement
        rel_imp = (best_val - loss_val)/best_val if best_val < float('inf') else float('inf')
        if rel_imp >= tol:
            best_val, no_imp = loss_val, 0
        else:
            no_imp += 1
        # early stop + rollback
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
    # 1) train/val split
    X_tr_np, X_val_np, Z_tr_np, Z_val_np, y_tr_np, y_val_np = train_test_split(
        X, Z, y, test_size=0.2, random_state=42
    )

    # 2) scale X
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np.astype(np.float32))
    X_val_np = scaler.transform(X_val_np.astype(np.float32))

    # 3) to tensors
    X_tr = torch.from_numpy(X_tr_np)
    Z_tr = torch.from_numpy(Z_tr_np.astype(np.float32))
    y_tr = torch.from_numpy(y_tr_np.astype(np.float32))
    X_val = torch.from_numpy(X_val_np)
    Z_val = torch.from_numpy(Z_val_np.astype(np.float32))
    y_val = torch.from_numpy(y_val_np.astype(np.float32))

    # 👈 Ensure y_tr and y_val are shaped (N,1)
    if y_tr.dim() == 1:
        y_tr = y_tr.unsqueeze(1)
    if y_val.dim() == 1:
        y_val = y_val.unsqueeze(1)

    # 4) model with 2 outputs
    input_dim = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2)
    )

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
        # — Training step —
        model.train()
        optimizer.zero_grad()
        out_tr = model(X_tr)               # shape [N,2]

        # 👈 Now repeat along the last dim to get (N,2)
        target_tr = y_tr.repeat(1, 2)
        mask_tr   = torch.cat([1 - Z_tr, Z_tr], dim=1)
        loss_tr = ((out_tr - target_tr)**2 * mask_tr).sum(dim=1).mean()
        loss_tr.backward()
        optimizer.step()
        train_losses.append(loss_tr.item())
        recent_states.append(copy.deepcopy(model.state_dict()))

        # — Validation step —
        model.eval()
        with torch.no_grad():
            out_val = model(X_val)
            target_val = y_val.repeat(1, 2)
            mask_val   = torch.cat([1 - Z_val, Z_val], dim=1)
            loss_val = ((out_val - target_val)**2 * mask_val).sum(dim=1).mean().item()

        val_losses.append(loss_val)
        scheduler.step(loss_val)

        # LR-change logging
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr != prev_lr:
            print(f"[Epoch {epoch}] LR changed: {prev_lr:.3e} → {curr_lr:.3e}")
            prev_lr = curr_lr

        # early-stopping / rollback logic
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
    **train_kwargs
):
    """
    X-learner with two tau-models:
      - Propensity e(x) by LogisticRegression
      - Outcome models mu0, mu1 via T-learner (optional reuse)
      - Pseudo-outcomes D0, D1 on controls/treated
      - Tau0(X): regressing D0 on X; Tau1(X): regressing D1 on X
      - Final tau_hat(x) = (1-e(x))*tau0(x) + e(x)*tau1(x)
    Returns:
      prop_model,
      (m0, scaler0), (m1, scaler1),
      tau0_model, scaler_tau0, tr0, val0, rollback0,
      tau1_model, scaler_tau1, tr1, val1, rollback1
    """
    # 1) Propensity
    prop_model = LogisticRegression().fit(X, Z.ravel())
    e = prop_model.predict_proba(X)[:, 1].reshape(-1, 1)

    # 2) (Re)compute T-learner outcome models
    if compute_t:
        m0, s0, _, _, m1, s1, _, _ = train_t_learner(X, Z, y, **train_kwargs)
    else:
        m0, s0, m1, s1 = t_models

    # 3) Get mu0(X), mu1(X) on *all* X
    X_np = X.astype(np.float32)
    mu0 = m0(torch.from_numpy(s0.transform(X_np))).detach().numpy().ravel()
    mu1 = m1(torch.from_numpy(s1.transform(X_np))).detach().numpy().ravel()

    # 4) Build pseudo-outcomes D0 (controls) and D1 (treated)
    mask0 = (Z.ravel() == 0)
    mask1 = (Z.ravel() == 1)

    D0 = (mu0[mask0] - y.ravel()[mask0]).reshape(-1, 1)
    X0 = X[mask0]

    D1 = (y.ravel()[mask1] - mu1[mask1]).reshape(-1, 1)
    X1 = X[mask1]

    # 5) Fit tau0 and tau1 separately
    tau0_model, scaler_tau0, tr0, val0, rb0 = train_group(X0, D0, **train_kwargs)
    print(f"X-learner τ0: parameters from epoch {rb0}")
    tau1_model, scaler_tau1, tr1, val1, rb1 = train_group(X1, D1, **train_kwargs)
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

    # reproducibility settings
    for sim in range(10):
        print(f"\n\nRun {sim}")
        SEED = 2025 + sim
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # data
        n_samples, n_features = 300, 1
        X = np.random.rand(n_samples, n_features).astype(np.float32)
        Z = np.random.binomial(1, 0.5, size=(n_samples,1)).astype(np.float32)
        y0 = (2*X + X*X).sum(axis=1,keepdims=True)
        y1 = (X + 3*X*X).sum(axis=1,keepdims=True)
        y = (1-Z)*y0 + Z*y1

        X_train, X_test, Z_train, Z_test, y_train, y_test, y0_t, y0_te, y1_t, y1_te = train_test_split(
            X, Z, y, y0, y1, test_size=0.3, random_state=SEED
        )

        # hyperparams
        params = dict(
            max_iter=300, tol=1e-3,
            hidden_dim=64, lr=0.01,
            patience=100, patience_lr=25,
            factor_lr=0.5
        )
        print("Training S, T, M, X learners...\n")
        # S
        m_s, sc_s, s_tr, s_val = train_s_learner(X_train, Z_train, y_train, **params)
        # T
        m0, sc0, t0_tr, t0_val, m1, sc1, t1_tr, t1_val = train_t_learner(X_train, Z_train, y_train, **params)
        # M
        m_m, sc_m, m_tr, m_val = train_m_learner(X_train, Z_train, y_train, **params)
        # X
        (
            prop_model,
            (m0, sc0), (m1, sc1),
            tau0_m, sc_tau0, x0_tr, x0_val, rb0,
            tau1_m, sc_tau1, x1_tr, x1_val, rb1
        ) = train_x_learner(
            X_train, Z_train, y_train, compute_t=False,
            t_models=(m0, sc0, m1, sc1), **params
        )

        # PEHE
        pehe = {}
        for name, est in [
            ('S', lambda x: (m_s(torch.from_numpy(sc_s.transform(np.concatenate([x,[[0.]],],axis=1).astype(np.float32)))).item(),
                              m_s(torch.from_numpy(sc_s.transform(np.concatenate([x,[[1.]],],axis=1).astype(np.float32)))).item())),
            ('T', lambda x: (m0(torch.from_numpy(sc0.transform(x.astype(np.float32)))).item(),
                              m1(torch.from_numpy(sc1.transform(x.astype(np.float32)))).item())),
            ('M', lambda x: tuple(
                                    m_m(torch.from_numpy(sc_m.transform(x.astype(np.float32)))).detach().numpy().ravel()
                            )
            ),
            ('X', lambda x: (
                            0.0, # the X learner return directly tau and not the 2 surface so we write tau as tau = tau - 0
                            (1 - prop_model.predict_proba(np.atleast_2d(x))[:, 1]) *
                            tau0_m(torch.from_numpy(sc_tau0.transform(np.atleast_2d(x)))).item() +
                            prop_model.predict_proba(np.atleast_2d(x))[:, 1] *
                            tau1_m(torch.from_numpy(sc_tau1.transform(np.atleast_2d(x)))).item()
                        ))
        ]:
            errs = []
            for i in range(len(X_test)):
                x_i = X_test[i:i+1]
                true_eff = (y1_te[i] - y0_te[i]).item()
                y0_hat, y1_hat = est(x_i)
                errs.append(( (y1_hat - y0_hat) - true_eff )**2)
            pehe[name] = math.sqrt(np.mean(errs))

        print("PEHE results:")
        for learner, val in pehe.items():
            print(f"{learner:>3s}-learner: {val:.6e}")

        # save curves
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
