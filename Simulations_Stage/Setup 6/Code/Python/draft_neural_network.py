import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Sample data
X = np.random.rand(10000, 10).astype(np.float32) # the more, the longer each epoch
                                                 # and the lower the convergence
                                                 # 10K seems a good trade off
y = (2 * X + X * X).sum(axis=1, keepdims=True).astype(np.float32)

# Split into training and testing sets
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2)

# Feature scaling
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Convert numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train_np)
y_train = torch.from_numpy(y_train_np)
X_test = torch.from_numpy(X_test_np)
y_test = torch.from_numpy(y_test_np)

# Define the model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# For logging
performance_log = []

# Training loop
max_iter = 3000
for epoch in range(max_iter):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Log every 50 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train).item()
            test_loss = criterion(model(X_test), y_test).item()
            performance_log.append([train_loss, test_loss])
            print(f"Epoch [{epoch+1}/{max_iter}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

# Store in pandas DataFrame
df_performance = pd.DataFrame(performance_log, columns=["Train Loss", "Test Loss"])
print("\nFinal Loss Log (Every 50 Epochs):")
print(df_performance)



