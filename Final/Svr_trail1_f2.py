import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('final_isotropic.csv')

df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Define predictors
feature_cols = ['Thickness_mm', 'Modulus_GPa', 'LoadLevel_kN', 'Subgrade_K_MPam'] + [col for col in df.columns if col.startswith("LoadPosition")]
X = df[feature_cols]
y = df[['Flexure_SR']]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and fit SVR model
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train.values.ravel())

# Predict
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# R² scores
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train R²: {r2_train:.4f}")
print(f"Test R²: {r2_test:.4f}")

# Plot results
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.scatter(y_train, y_train_pred, color='red')
plt.plot(y_train, y_train, 'k--')
plt.title(f'Flexural Stress Ratio(Train) R² = {r2_train:.4f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1,2,2)
plt.scatter(y_test, y_test_pred, color='red')
plt.plot(y_test, y_test, 'k--')
plt.title(f'Flexural Stress Ratio(Test) R² = {r2_test:.4f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.suptitle('SVR Prediction of Flexural Stress Ratio', fontsize=16, y=1.05)
plt.show()
