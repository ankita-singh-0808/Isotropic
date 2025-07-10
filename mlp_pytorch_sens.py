import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import random

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------
# 1. Load and preprocess data
# --------------------------------------------
df = pd.read_csv('final_isotropic_augmented.csv')
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

X = df.drop('Flexure_SR', axis=1).values
y = df['Flexure_SR'].values.reshape(-1, 1)
feature_names = df.drop('Flexure_SR', axis=1).columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# --------------------------------------------
# 2. Dataset for PyTorch
# --------------------------------------------
class FlexureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FlexureDataset(X_train, y_train), batch_size=32, shuffle=True)

# --------------------------------------------
# 3. MLP Model
# --------------------------------------------
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

model = MLP(input_dim=X.shape[1]).to(device)

# --------------------------------------------
# 4. Training
# --------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1000
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

# --------------------------------------------
# 5. Evaluation + R² Scores
# --------------------------------------------
model.eval()
with torch.no_grad():
    # Test
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_test_scaled = model(X_test_tensor).cpu().numpy()
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
    y_true_test = scaler_y.inverse_transform(y_test)

    # Train
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_pred_train_scaled = model(X_train_tensor).cpu().numpy()
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled)
    y_true_train = scaler_y.inverse_transform(y_train)

r2_train = r2_score(y_true_train, y_pred_train)
r2_test = r2_score(y_true_test, y_pred_test)

print(f"\nTrain R²: {r2_train:.4f}")
print(f"Test R²: {r2_test:.4f}")

# --------------------------------------------
# 6. Plot: Train and Test Predictions
# --------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

axs[0].scatter(y_true_train, y_pred_train, alpha=0.6, color='green', label='Training')
axs[0].plot([y_true_train.min(), y_true_train.max()], [y_true_train.min(), y_true_train.max()], 'r--', label='Reference Line')
axs[0].set_title(f'Training: Actual vs Predicted\nR² = {r2_train:.4f}')
axs[0].set_xlabel('Actual Flexure_SR')
axs[0].set_ylabel('Predicted Flexure_SR')
axs[0].grid(True)
axs[0].legend()

axs[1].scatter(y_true_test, y_pred_test, alpha=0.6, color='blue', label='Testing')
axs[1].plot([y_true_test.min(), y_true_test.max()], [y_true_test.min(), y_true_test.max()], 'r--', label='Reference Line')
axs[1].set_title(f'Testing: Actual vs Predicted\nR² = {r2_test:.4f}')
axs[1].set_xlabel('Actual Flexure_SR')
axs[1].set_ylabel('Predicted Flexure_SR')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

# --------------------------------------------
# 7. K-Fold Cross Validation with R² (Train + Test)
# --------------------------------------------
def run_kfold_cv(model_class, X, y, n_splits=5, epochs=1000, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_train_scores, r2_val_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train_fold)
        X_val_scaled = scaler_X.transform(X_val_fold)
        y_train_scaled = scaler_y.fit_transform(y_train_fold)
        y_val_scaled = scaler_y.transform(y_val_fold)

        train_loader = DataLoader(FlexureDataset(X_train_scaled, y_train_scaled), batch_size=batch_size, shuffle=True)

        model = model_class(input_dim=X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            # Train R²
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_pred = model(X_train_tensor).cpu().numpy()
            y_train_pred = scaler_y.inverse_transform(y_train_pred)
            y_train_true = scaler_y.inverse_transform(y_train_scaled)
            r2_train = r2_score(y_train_true, y_train_pred)
            r2_train_scores.append(r2_train)

            # Test R²
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            y_val_pred = model(X_val_tensor).cpu().numpy()
            y_val_pred = scaler_y.inverse_transform(y_val_pred)
            y_val_true = scaler_y.inverse_transform(y_val_scaled)
            r2_val = r2_score(y_val_true, y_val_pred)
            r2_val_scores.append(r2_val)

    print(f"\n✅ Average Train R²: {np.mean(r2_train_scores):.4f} ± {np.std(r2_train_scores):.4f}")
    print(f"✅ Average Test R²:  {np.mean(r2_val_scores):.4f} ± {np.std(r2_val_scores):.4f}")

    return r2_train_scores, r2_val_scores

# Run K-Fold
train_r2_list, test_r2_list = run_kfold_cv(MLP, X, y)

# --------------------------------------------
# 8. Linear Sensitivity Plot
# --------------------------------------------
def plot_linear_sensitivity(model, X_scaled, feature_names, scaler_X, scaler_y, steps=50):
    model.eval()
    X_unscaled = scaler_X.inverse_transform(X_scaled)
    X_mean_original = np.mean(X_unscaled, axis=0)

    # 🔥 Filter out dummy features like LoadPosition_EL
    filtered_features = [fname for fname in feature_names if not fname.startswith("LoadPosition_")]
    filtered_indices = [feature_names.index(fname) for fname in filtered_features]

    n_features = len(filtered_features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    with torch.no_grad():
        for plot_idx, feature_idx in enumerate(filtered_indices):
            fname = feature_names[feature_idx]
            col_min = X_unscaled[:, feature_idx].min()
            col_max = X_unscaled[:, feature_idx].max()
            val_range = np.linspace(col_min, col_max, steps)
            predictions = []

            for val in val_range:
                X_temp = X_mean_original.copy()
                X_temp[feature_idx] = val
                X_scaled_temp = scaler_X.transform(X_temp.reshape(1, -1))
                X_tensor = torch.tensor(X_scaled_temp, dtype=torch.float32).to(device)
                y_pred = model(X_tensor).cpu().numpy()
                y_original = scaler_y.inverse_transform(y_pred)[0][0]
                predictions.append(y_original)

            axes[plot_idx].plot(val_range, predictions, marker='o', color='indigo')
            axes[plot_idx].set_title(f'{fname}')
            axes[plot_idx].set_xlabel(fname)
            axes[plot_idx].set_ylabel('Predicted Flexure_SR')
            axes[plot_idx].grid(True)

        for j in range(plot_idx + 1, len(axes)):
            axes[j].axis('off')

    plt.suptitle('MLP: Sensitivity Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Plot sensitivity
plot_linear_sensitivity(model, X_test, feature_names, scaler_X, scaler_y)
