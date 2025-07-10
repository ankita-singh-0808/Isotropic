import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('final_isotropic_augmented.csv')

# Define features and target
features = ['Thickness_mm', 'Modulus_GPa', 'LoadLevel_kN', 'Subgrade_K_MPam', 'LoadPosition']
target = 'Flexure_SR'

X = df[features]
y = df[target].values

# Preprocessing
numerical_features = ['Thickness_mm', 'Modulus_GPa', 'LoadLevel_kN', 'Subgrade_K_MPam']
categorical_features = ['LoadPosition']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=40)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train.toarray() if hasattr(X_train, 'toarray') else X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray() if hasattr(X_test, 'toarray') else X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Dataset class
class FlexureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader
train_loader = DataLoader(FlexureDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(FlexureDataset(X_test_tensor, y_test_tensor), batch_size=32)

# MLP Model
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

# Instantiate model, loss, optimizer
model = MLPRegressor(input_dim=X_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor).numpy().flatten()
    test_preds = model(X_test_tensor).numpy().flatten()

r2_train = r2_score(y_train, train_preds)
r2_test = r2_score(y_test, test_preds)
mse_test = mean_squared_error(y_test, test_preds)

print(f"Train R²: {r2_train:.4f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Plotting predictions
plt.figure(figsize=(12, 5))

# Training data plot
plt.subplot(1, 2, 1)
plt.scatter(y_train, train_preds, alpha=0.7, color='blue', label='Training')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', label='Reference Line')
plt.xlabel('Actual Flexure_SR')
plt.ylabel('Predicted Flexure_SRa')
plt.title(f'Training Data\n$R^2 = {r2_train:.4f}$')
plt.grid(True)
plt.legend()

# Testing data plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_preds, alpha=0.7, color='green', label='Testing')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Reference Line')
plt.xlabel('Actual Flexure_SR')
plt.ylabel('Predicted Flexure_SR')
plt.title(f'Testing Data\n$R^2 = {r2_test:.4f}$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
