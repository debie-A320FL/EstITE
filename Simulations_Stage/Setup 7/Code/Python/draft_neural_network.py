import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Sample size and features
n_samples = 10000
n_features = 1

# Covariates
X = np.random.rand(n_samples, n_features).astype(np.float32)

# Binary treatment variable
Z = np.random.binomial(1, 0.5, size=(n_samples, 1)).astype(np.float32)

# Generate y based on treatment group
y0 = (2 * X + X * X).sum(axis=1, keepdims=True)
y1 = (X + 3 * X * X).sum(axis=1, keepdims=True)
y = (1 - Z) * y0 + Z * y1  # Choose y0 if Z=0, y1 if Z=1

# Split data based on Z
X0 = X[Z[:, 0] == 0]
y0 = y[Z[:, 0] == 0]

X1 = X[Z[:, 0] == 1]
y1 = y[Z[:, 0] == 1]

def train_model(X_data, y_data, label, max_iter=1000, log_every=50):
    # Train-test split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_data, y_data, test_size=0.2)

    # Scaling
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)

    # Convert to tensors
    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    X_test = torch.from_numpy(X_test_np)
    y_test = torch.from_numpy(y_test_np)

    # Define model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    performance_log = []

    # Training loop
    for epoch in range(max_iter):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % log_every == 0:
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(X_train), y_train).item()
                test_loss = criterion(model(X_test), y_test).item()
                performance_log.append([train_loss, test_loss])
                print(f"[Z={label}] Epoch [{epoch+1}/{max_iter}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    return pd.DataFrame(performance_log, columns=["Train Loss", "Test Loss"])

# Train a single model
print("\nTraining model for Z=0 & 1...") # it's suck at after 600 epoch 0.03 => bad idea when both surface are different
df_perf_01 = train_model(X, y, label="0&1")

# Train separate models
print("\nTraining model for Z=0...")
df_perf_0 = train_model(X0, y0, label=0)

print("\nTraining model for Z=1...")
df_perf_1 = train_model(X1, y1, label=1)

# Quicker convergence, and continue to decrease until the end

# Show performance logs
print("\nFinal Loss Log (Z=0&1):")
print(df_perf_01.iloc[-1])

print("\nFinal Loss Log (Z=0):")
print(df_perf_0.iloc[-1])

print("\nFinal Loss Log (Z=1):")
print(df_perf_1.iloc[-1])
