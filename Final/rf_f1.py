import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
file_path = 'final_isotropic_augmented.csv'  # Replace with your actual file
df = pd.read_csv(file_path)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Features and target
X = df.drop(['Flexure_SR'], axis=1)
y = df['Flexure_SR']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train RF Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = rf.predict(X_train_scaled)
y_test_pred = rf.predict(X_test_scaled)

# Metrics
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Training MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
print(f"Testing MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# Plot: Actual vs Predicted (Training and Testing)
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Training plot
axs[0].scatter(y_train, y_train_pred, alpha=0.6, color='green', label='Training')
axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Reference Line')
axs[0].set_title(f'Training: Actual vs Predicted\nR² = {r2_train:.4f}')
axs[0].set_xlabel('Actual Flexure_SR')
axs[0].set_ylabel('Predicted Flexure_SR')
axs[0].grid(True)
axs[0].legend()

# Testing plot
axs[1].scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Testing')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Reference Line')
axs[1].set_title(f'Testing: Actual vs Predicted\nR² = {r2_test:.4f}')
axs[1].set_xlabel('Actual Flexure_SR')
axs[1].set_ylabel('Predicted Flexure_SR')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
