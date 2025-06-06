def train_s_learner(
    X, Z, y,
    max_iter=1000,
    tol=1e-3,
    patience=10,
    hidden_dim=64,
    lr=0.001
):
    """
    S‐learner with “roll back” patience:
      - Stops only when rel_imp < tol for `patience` consecutive epochs.
      - When stopping, returns the model from `patience` epochs ago.
    Returns:
      - model: the rolled‐back model
      - scaler: fitted StandardScaler
    """

    # 1) Combine X and Z
    X_with_Z = np.concatenate([X, Z], axis=1)

    # 2) Train/val split (fixed random_state recommended)
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_with_Z, y,
        test_size=0.2,
        random_state=42
    )

    # 3) Scale features
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np.astype(np.float32))
    X_val_np = scaler.transform(X_val_np.astype(np.float32))

    # 4) To tensors
    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np.astype(np.float32))
    X_val = torch.from_numpy(X_val_np)
    y_val = torch.from_numpy(y_val_np.astype(np.float32))

    # 5) Build model
    input_dim = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    prev_val_loss = float('inf')
    no_improve_count = 0

    # Keep the last `patience` state_dicts in a deque
    recent_states = deque(maxlen=patience)

    for epoch in range(1, max_iter + 1):
        # ---- (A) Training step ----
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        train_loss = criterion(preds, y_train)
        train_loss.backward()
        optimizer.step()

        # Save a copy of this epoch’s model parameters at the end of the epoch
        # We’ll later roll back to one of these if needed.
        recent_states.append(copy.deepcopy(model.state_dict()))

        # ---- (B) Validation step ----
        model.eval()
        with torch.no_grad():
            current_val_loss = criterion(model(X_val), y_val).item()

        # ---- (C) Check relative improvement ----
        if prev_val_loss < float('inf'):
            rel_imp = (prev_val_loss - current_val_loss) / prev_val_loss
        else:
            # For the very first epoch, force rel_imp to be large so no “early stop” immediately
            rel_imp = float('inf')

        if rel_imp < tol:
            no_improve_count += 1
        else:
            no_improve_count = 0
            prev_val_loss = current_val_loss


        # ---- (D) If we’ve failed to improve `patience` times in a row → stop & roll back
        if no_improve_count >= patience:
            # The oldest state in recent_states is from `patience` epochs ago.
            rollback_state = recent_states[0]
            model.load_state_dict(rollback_state)
            print(
                f"S‐learner: stopping at epoch {epoch} "
                f"after {patience} epochs with rel_imp < {tol:.6e}. "
                f"Rolling back to epoch {epoch-patience} model."
            )
            break

        if epoch == max_iter:
            print(f"Reaching epoch {epoch} with a relative improvement of {rel_imp:.6f}")

    return model, scaler





def train_t_learner(
    X, Z, y,
    max_iter=1000,
    tol=1e-3,
    patience=10,
    hidden_dim=64,
    lr=0.001
):
    """
    T‐learner with roll‐back patience:
      - Trains two separate models for Z=0 and Z=1
      - Each has its own early‐stopping + roll‐back after `patience` bad epochs
    Returns:
      - model0, scaler0 (for Z=0)
      - model1, scaler1 (for Z=1)
    """

    # Split data by treatment
    X0 = X[Z[:, 0] == 0]
    y0 = y[Z[:, 0] == 0]
    X1 = X[Z[:, 0] == 1]
    y1 = y[Z[:, 0] == 1]

    def train_group(X_group, y_group, label):
        # 1) Train/val split
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_group, y_group,
            test_size=0.2,
            random_state=42
        )
        # 2) Scale
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train_np.astype(np.float32))
        X_val_np = scaler.transform(X_val_np.astype(np.float32))

        # 3) To torch tensors
        X_train = torch.from_numpy(X_train_np)
        y_train = torch.from_numpy(y_train_np.astype(np.float32))
        X_val = torch.from_numpy(X_val_np)
        y_val = torch.from_numpy(y_val_np.astype(np.float32))

        # 4) Build model
        input_dim = X_train.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        prev_val_loss = float('inf')
        no_improve_count = 0
        recent_states = deque(maxlen=patience)

        for epoch in range(1, max_iter + 1):
            # ---- Training ----
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            train_loss = criterion(preds, y_train)
            train_loss.backward()
            optimizer.step()

            # Save current state
            recent_states.append(copy.deepcopy(model.state_dict()))

            # ---- Validation ----
            model.eval()
            with torch.no_grad():
                current_val_loss = criterion(model(X_val), y_val).item()

            if prev_val_loss < float('inf'):
                rel_imp = (prev_val_loss - current_val_loss) / prev_val_loss
            else:
                rel_imp = float('inf')

            if rel_imp < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
                prev_val_loss = current_val_loss

            if no_improve_count >= patience:
                rollback_state = recent_states[0]
                model.load_state_dict(rollback_state)
                print(
                    f"T‐learner (Z={label}): stopping at epoch {epoch} "
                    f"after {patience} bad epochs. Rolling back to epoch {epoch-patience}."
                )
                break

            if epoch == max_iter:
                print(f"Reaching epoch {epoch} with a relative improvement of {rel_imp:.6f}")

        return model, scaler

    model0, scaler0 = train_group(X0, y0, label=0)
    model1, scaler1 = train_group(X1, y1, label=1)
    return model0, scaler0, model1, scaler1



if __name__ == "__main__":
    # === PEHE evaluation on a larger sample ===
    import math
    import os
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import copy
    from collections import deque

    # 1.1 Pick a single integer “seed” for reproducibility:
    SEED = 2025

    # 1.2 Python built‐ins:
    random.seed(SEED)

    # 1.3 NumPy
    np.random.seed(SEED)

    # 1.4 PyTorch (both CPU and GPU, if you ever switch to a GPU)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 1.5 Make PyTorch’s CuDNN deterministic (only relevant if you use a GPU)
    #     and disable the benchmark mode so it doesn’t pick new kernels at random:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1) Generate a larger dataset and true potential outcomes
    n_samples_big = 10000
    n_features = 1

    X_big = np.random.rand(n_samples_big, n_features).astype(np.float32)
    Z_big = np.random.binomial(1, 0.5, size=(n_samples_big, 1)).astype(np.float32)

    y0_big = (2 * X_big + X_big * X_big).sum(axis=1, keepdims=True)
    y1_big = (X_big + 3 * X_big * X_big).sum(axis=1, keepdims=True)
    y_big = (1 - Z_big) * y0_big + Z_big * y1_big

    # 2) Split into train/test (80/20), keeping y0 and y1 for PEHE
    X_train, X_test, Z_train, Z_test, y_train, y_test, y0_train, y0_test, y1_train, y1_test = train_test_split(
        X_big, Z_big, y_big, y0_big, y1_big, test_size=0.3, random_state=SEED
    )

    # 3) Train S-learner and T-learner on the training set
    tol = 1e-3
    max_iter = 10000
    hidden_dim = 64
    lr = 0.001
    patience = 20
    print("starting...")
    model_s, scaler_s = train_s_learner(X_train, Z_train, y_train,
                                        max_iter=max_iter, tol=tol,
                                        hidden_dim=hidden_dim, lr=lr,
                                        patience=patience)

    model0_t, scaler0_t, model1_t, scaler1_t = train_t_learner(X_train, Z_train, y_train,
                                                                max_iter=max_iter, tol=tol,
                                                                hidden_dim=hidden_dim, lr=lr,
                                                                patience=patience)

    # 4) On the test set, compute PEHE for each learner
    pehe_s_list = []
    pehe_t_list = []

    for i in range(len(X_test)):
        x_i = X_test[i : i + 1]            # shape (1, n_features)
        true_effect = (y1_test[i] - y0_test[i]).item()

        # S-learner: estimate y_hat(z=0) and y_hat(z=1)
        xz0 = np.concatenate([x_i, np.array([[0.0]], dtype=np.float32)], axis=1)
        xz1 = np.concatenate([x_i, np.array([[1.0]], dtype=np.float32)], axis=1)

        xz0_scaled = scaler_s.transform(xz0)
        xz1_scaled = scaler_s.transform(xz1)

        with torch.no_grad():
            y0_hat_s = model_s(torch.from_numpy(xz0_scaled)).item()
            y1_hat_s = model_s(torch.from_numpy(xz1_scaled)).item()

        est_effect_s = y1_hat_s - y0_hat_s

        # T-learner: estimate with separate models (ignore Z)
        x0_scaled = scaler0_t.transform(x_i)
        x1_scaled = scaler1_t.transform(x_i)

        with torch.no_grad():
            y0_hat_t = model0_t(torch.from_numpy(x0_scaled)).item()
            y1_hat_t = model1_t(torch.from_numpy(x1_scaled)).item()

        est_effect_t = y1_hat_t - y0_hat_t

        pehe_s_list.append((est_effect_s - true_effect) ** 2)
        pehe_t_list.append((est_effect_t - true_effect) ** 2)

    pehe_s = math.sqrt(np.mean(pehe_s_list))
    pehe_t = math.sqrt(np.mean(pehe_t_list))

    print(f"PEHE (S-learner): {pehe_s:.6f}")
    print(f"PEHE (T-learner): {pehe_t:.6f}")