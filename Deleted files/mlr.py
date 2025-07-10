import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'final_isotropic.csv'  # Replace with your actual CSV path
df = pd.read_csv(file_path)

# Drop rows with any missing values
df.dropna(inplace=True)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Feature and target separation
X = df.drop(['Flexure_SR'], axis=1)
y = df['Flexure_SR']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Multiple Linear Regression model
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)

# Predict
y_train_pred = mlr.predict(X_train_scaled)
y_test_pred = mlr.predict(X_test_scaled)

# Metrics
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Training MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
print(f"Testing MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# Plotting Actual vs Predicted for Training and Testing
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Training plot
axs[0].scatter(y_train, y_train_pred, color='green', alpha=0.6, label='Train')
axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Ideal Fit')
axs[0].set_title(f'Training: Actual vs Predicted\nR² = {r2_train:.4f}')
axs[0].set_xlabel('Actual Flexure_SR')
axs[0].set_ylabel('Predicted Flexure_SR')
axs[0].grid(True)
axs[0].legend()

# Testing plot
axs[1].scatter(y_test, y_test_pred, color='blue', alpha=0.6, label='Test')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
axs[1].set_title(f'Testing: Actual vs Predicted\nR² = {r2_test:.4f}')
axs[1].set_xlabel('Actual Flexure_SR')
axs[1].set_ylabel('Predicted Flexure_SR')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
