import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('final_isotropic.csv')

# One-hot encode
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Features and target
# drop_cols = ['Max_Flexure_MPa', 'Max_Shear_MPa', 'Compressive_Strength_Mpa', 'Flexural Strength_Mpa']
X = df.drop('Flexure_SR', axis=1)
y = df['Flexure_SR'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

model = MLP(X_train_tensor.shape[1])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 200
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 20 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            train_pred = model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor).item()
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy().flatten()
    y_test_pred = model(X_test_tensor).numpy().flatten()

# R² and MSE
print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test  R²: {r2_score(y_test, y_test_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")

mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Plot: Actual vs Predicted (Training and Testing)
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Training plot
axs[0].scatter(y_train, y_train_pred, alpha=0.6, color='green', label='Train Data')
axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Ideal Fit')
axs[0].set_title(f'Training: Actual vs Predicted\nR² = {r2_train:.4f}')
axs[0].set_xlabel('Actual Flexure_SR')
axs[0].set_ylabel('Predicted Flexure_SR')
axs[0].grid(True)
axs[0].legend()

# Testing plot
axs[1].scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Test Data')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
axs[1].set_title(f'Testing: Actual vs Predicted\nR² = {r2_test:.4f}')
axs[1].set_xlabel('Actual Flexure_SR')
axs[1].set_ylabel('Predicted Flexure_SR')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
