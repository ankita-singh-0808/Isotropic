import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('final_isotropic.csv')

# Features and target
features = ['Thickness_mm', 'Modulus_GPa', 'LoadLevel_kN', 'Subgrade_K_MPam', 'LoadPosition']
target = 'Flexure_SR'

X = df[features]
y = df[target]

# Separate numerical and categorical features
numerical_features = ['Thickness_mm', 'Modulus_GPa', 'LoadLevel_kN', 'Subgrade_K_MPam']
categorical_features = ['LoadPosition']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# MLP Regressor pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50),
                               activation='relu',
                               solver='adam',
                               max_iter=1000,
                               random_state=40))])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions
train_preds = pipeline.predict(X_train)
test_preds = pipeline.predict(X_test)

# Evaluation
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
plt.scatter(y_train, train_preds, alpha=0.7, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual Flexure_SR')
plt.ylabel('Predicted Flexure_SR')
plt.title(f'Training Data\n$R^2 = {r2_train:.4f}$')
plt.grid(True)

# Testing data plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_preds, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Flexure_SR')
plt.ylabel('Predicted Flexure_SR')
plt.title(f'Testing Data\n$R^2 = {r2_test:.4f}$')
plt.grid(True)

plt.tight_layout()
plt.show()
