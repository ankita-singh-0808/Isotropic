import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
file_path = 'final_isotropic.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Features and target
X = df.drop(['Flexure_SR'], axis=1)
y = df['Flexure_SR']

# Define pipeline: Scaling + Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Define K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated R² and MSE scores
r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
mse_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')

print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")

# Get cross-validated predictions for plotting
y_pred = cross_val_predict(pipeline, X, y, cv=kf)
overall_r2 = r2_score(y, y_pred)

# Plot: Actual vs Predicted (Cross-Validated)
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, color='purple', label='Cross-Validated Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
plt.title(f'Cross-Validated: Actual vs Predicted\nR² = {overall_r2:.4f}')
plt.xlabel('Actual Flexure_SR')
plt.ylabel('Predicted Flexure_SR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot R² and MSE scores for each fold (fixing Seaborn warnings)
folds = np.arange(1, len(r2_scores) + 1)
fold_labels = [f'Fold {i}' for i in folds]

plt.figure(figsize=(14, 6))

# R² plot
plt.subplot(1, 2, 1)
sns.barplot(x=fold_labels, y=r2_scores, palette='Greens_r')
plt.axhline(np.mean(r2_scores), color='red', linestyle='--', label=f'Mean R² = {np.mean(r2_scores):.4f}')
plt.title('R² Score per Fold')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.ylim(-1, 1)  # Adjust to show negative values if needed
plt.legend()
plt.grid(True)

# MSE plot
plt.subplot(1, 2, 2)
sns.barplot(x=fold_labels, y=mse_scores, palette='Blues_r')
plt.axhline(np.mean(mse_scores), color='red', linestyle='--', label=f'Mean MSE = {np.mean(mse_scores):.4f}')
plt.title('MSE per Fold')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
