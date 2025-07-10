import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV data
file_path = 'final_isotropic_augmented.csv'
df = pd.read_csv(file_path)

# Drop 'Case' if it seems less useful; keep only LoadPosition
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Select relevant features explicitly (feel free to adjust)
selected_features = ['Thickness_mm', 'Modulus_GPa', 'LoadLevel_kN', 'Subgrade_K_MPam'] + \
                    [col for col in df.columns if col.startswith('LoadPosition')]

X = df[selected_features]
y = df['Flexure_SR']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features with MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and fit SVR model with initial parameters
# svr = SVR(kernel='rbf', C=10, epsilon=0.2, gamma='scale')    # without this results are better
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)

# Predict
y_train_pred = svr.predict(X_train_scaled)
y_test_pred = svr.predict(X_test_scaled)

# Metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Training MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
print(f"Testing  MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Training Plot
axs[0].scatter(y_train, y_train_pred, color='green', alpha=0.6, label='Training')
axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='Reference Line')
axs[0].set_title(f'Training Set\nR² = {r2_train:.4f}')
axs[0].set_xlabel('Actual Flexural Stress Ratio (MPa)')
axs[0].set_ylabel('Predicted')
axs[0].grid(True)
axs[0].legend()

# Testing Plot
axs[1].scatter(y_test, y_test_pred, color='blue', alpha=0.6, label='Testing')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Reference Line')
axs[1].set_title(f'Testing Set\nR² = {r2_test:.4f}')
axs[1].set_xlabel('Actual Flexural Stress Ratio (MPa)')
axs[1].set_ylabel('Predicted')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
