def train_s_learner(
    X, Z, y,
    max_iter=1000,
    tol=1e-3,
    patience=10,
    hidden_dim=64,
    lr=0.001,
    patience_lr = 20,
    factor_lr = 0.25
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

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=factor_lr, patience=patience_lr, cooldown=2* patience_lr
    )

    # Initial learning rate tracking
    prev_lr = optimizer.param_groups[0]['lr']

    train_losses = []
    val_losses   = []

    best_val_loss = float('inf')
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
        train_losses.append(train_loss.item())

        # Save a copy of this epoch’s model parameters at the end of the epoch
        # We’ll later roll back to one of these if needed.
        recent_states.append(copy.deepcopy(model.state_dict()))

        # ---- (B) Validation step ----
        model.eval()
        with torch.no_grad():
            current_val_loss = criterion(model(X_val), y_val).item()

        scheduler.step(current_val_loss)

        # Check if LR changed
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"[Epoch {epoch}] LR changed: {prev_lr:.3e} → {current_lr:.3e}")
            prev_lr = current_lr
            
        val_losses.append(current_val_loss)

        # ---- (C) Check relative improvement ----
        if best_val_loss < float('inf'):
            rel_imp = (best_val_loss - current_val_loss) / best_val_loss
        else:
            rel_imp = float('inf')

        if rel_imp >= tol:
            best_val_loss = current_val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1


        # ---- (D) If we’ve failed to improve `patience` times in a row → stop & roll back
        if no_improve_count >= patience:
            # The oldest state in recent_states is from `patience` epochs ago.
            rollback_state = recent_states[0]
            model.load_state_dict(rollback_state)
            print(
                f"S‐learner: stopping at epoch {epoch} "
                f"after {patience} epochs with {rel_imp:.3e} < {tol:.3e}. "
                f"Rolling back to epoch {epoch-patience} model."
            )
            break

        if epoch == max_iter:
            print(f"Reaching epoch {epoch} with a relative improvement of {rel_imp:.3e}")

    return model, scaler,train_losses, val_losses





def train_t_learner(
    X, Z, y,
    max_iter=100,
    tol=1e-3,
    patience=10,
    hidden_dim=64,
    lr=0.001,
    patience_lr = 20,
    factor_lr = 0.25
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

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=factor_lr, patience=patience_lr, cooldown=2* patience_lr
        )

        # Initial learning rate tracking
        prev_lr = optimizer.param_groups[0]['lr']

        train_losses = []
        val_losses   = []

        best_val_loss = float('inf')
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
            train_losses.append(train_loss.item())

            # Save current state
            recent_states.append(copy.deepcopy(model.state_dict()))

            # ---- Validation ----
            model.eval()
            with torch.no_grad():
                current_val_loss = criterion(model(X_val), y_val).item()
            
            scheduler.step(current_val_loss)

            # Check if LR changed
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                print(f"[Epoch {epoch}] LR changed: {prev_lr:.3e} → {current_lr:.3e}")
                prev_lr = current_lr

            val_losses.append(current_val_loss)

            if best_val_loss < float('inf'):
                rel_imp = (best_val_loss - current_val_loss) / best_val_loss
            else:
                rel_imp = float('inf')

            if rel_imp >= tol:
                best_val_loss = current_val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1


            if no_improve_count >= patience:
                rollback_state = recent_states[0]
                model.load_state_dict(rollback_state)
                print(
                    f"T‐learner (Z={label}): stopping at epoch {epoch} "
                    f"after {patience} epochs with {rel_imp:.3e} < {tol:.3e}. "
                    f"Rolling back to epoch {epoch-patience} model."
                )
                break

            if epoch == max_iter:
                print(f"Reaching epoch {epoch} with a relative improvement of {rel_imp:.3e}")

        return model, scaler,train_losses, val_losses

    model0, scaler0, train_losses0, val_losses0 = train_group(X0, y0, label=0)
    model1, scaler1, train_losses1, val_losses1 = train_group(X1, y1, label=1)
    return model0, scaler0, train_losses0, val_losses0, model1, scaler1, train_losses1, val_losses1


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
    from torch.optim.lr_scheduler import ReduceLROnPlateau


    for sim in range(10):
        print(f"\n\nRun number {sim}")
        # 1.1 Pick a single integer “seed” for reproducibility:
        SEED = 2025 + sim

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
        Z_big = np.random.binomial(1, 0.9, size=(n_samples_big, 1)).astype(np.float32)

        y0_big = (2 * X_big + X_big * X_big).sum(axis=1, keepdims=True)
        y1_big = (X_big + 3 * X_big * X_big).sum(axis=1, keepdims=True)
        y_big = (1 - Z_big) * y0_big + Z_big * y1_big

        # 2) Split into train/test (80/20), keeping y0 and y1 for PEHE
        X_train, X_test, Z_train, Z_test, y_train, y_test, y0_train, y0_test, y1_train, y1_test = train_test_split(
            X_big, Z_big, y_big, y0_big, y1_big, test_size=0.3, random_state=SEED
        )

        # 3) Train S-learner and T-learner on the training set
        tol = 1e-3
        max_iter = 15000
        hidden_dim = 64
        lr = 0.01
        patience = 100
        patience_lr = 25
        factor_lr = 0.5
        print("starting...")
        model_s, scaler_s, s_train, s_val = train_s_learner(X_train, Z_train, y_train,
                                            max_iter=max_iter,
                                            tol=tol,
                                            hidden_dim=hidden_dim,
                                            lr=lr,
                                            patience=patience,
                                            patience_lr=patience_lr,
                                            factor_lr=factor_lr)

        (model0, scaler0, t0_train, t0_val,
        model1, scaler1, t1_train, t1_val) = train_t_learner(
                                                    X_train, Z_train, y_train,
                                                    max_iter=max_iter,
                                                    tol=tol,
                                                    hidden_dim=hidden_dim,
                                                    lr=lr,
                                                    patience=patience,
                                                    patience_lr=patience_lr,
                                                    factor_lr=factor_lr)

        # 4) On the test set, compute PEHE for each learner
        pehe_s_list = []
        pehe_t_list = []

        for i in range(len(X_test)):
            x_i = X_test[i : i + 1]            # shape (1, n_features)s
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
            x0_scaled = scaler0.transform(x_i)
            x1_scaled = scaler1.transform(x_i)

            with torch.no_grad():
                y0_hat_t = model0(torch.from_numpy(x0_scaled)).item()
                y1_hat_t = model1(torch.from_numpy(x1_scaled)).item()

            est_effect_t = y1_hat_t - y0_hat_t

            pehe_s_list.append((est_effect_s - true_effect) ** 2)
            pehe_t_list.append((est_effect_t - true_effect) ** 2)

        pehe_s = math.sqrt(np.mean(pehe_s_list))
        pehe_t = math.sqrt(np.mean(pehe_t_list))

        print(f"PEHE (S-learner): {pehe_s:.6f}")
        print(f"PEHE (T-learner): {pehe_t:.6f}")

        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        with PdfPages(f'learning_curves_{sim}_patience_{patience}_lr_cooldown_bis.pdf') as pdf:
            # S-learner
            plt.figure()
            plt.plot(s_train, label='Train Loss')
            plt.plot(s_val,   label='Validation Loss')
            plt.yscale('log')
            plt.title('S-learner Learning Curve')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            pdf.savefig(); plt.close()

            # T-learner Z=0
            plt.figure()
            plt.plot(t0_train, label='Train Loss')
            plt.plot(t0_val,   label='Validation Loss')
            plt.yscale('log')
            plt.title('T-learner (Z=0) Learning Curve')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            pdf.savefig(); plt.close()

            # T-learner Z=1
            plt.figure()
            plt.plot(t1_train, label='Train Loss')
            plt.plot(t1_val,   label='Validation Loss')
            plt.yscale('log')
            plt.title('T-learner (Z=1) Learning Curve')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            pdf.savefig(); plt.close()

        print("Saved learning curves to learning_curves.pdf")
