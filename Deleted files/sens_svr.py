import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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

# Define pipeline: Scaling + SVR
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
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

# Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, color='purple', label='Cross-Validated Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
plt.title(f'SVR: Actual vs Predicted\nR² = {overall_r2:.4f}')
plt.xlabel('Actual Flexure_SR')
plt.ylabel('Predicted Flexure_SR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot R² and MSE scores per fold
folds = np.arange(1, len(r2_scores) + 1)
fold_labels = [f'Fold {i}' for i in folds]

plt.figure(figsize=(14, 6))

# R² bar plot
plt.subplot(1, 2, 1)
sns.barplot(x=fold_labels, y=r2_scores, hue=fold_labels, palette='Purples', legend=False)
plt.axhline(np.mean(r2_scores), color='red', linestyle='--', label=f'Mean R² = {np.mean(r2_scores):.4f}')
plt.title('SVR: R² Score per Fold')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.ylim(-1, 1)
plt.legend()
plt.grid(True)

# MSE bar plot
plt.subplot(1, 2, 2)
sns.barplot(x=fold_labels, y=mse_scores, hue=fold_labels, palette='BuPu', legend=False)
plt.axhline(np.mean(mse_scores), color='red', linestyle='--', label=f'Mean MSE = {np.mean(mse_scores):.4f}')
plt.title('SVR: MSE per Fold')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# -------------------------------
# ✅ Sensitivity Analysis Function
# -------------------------------
def sensitivity_analysis(pipeline, X, feature_names, percent_change=0.1):
    base_preds = pipeline.predict(X)
    sensitivities = []

    for col in feature_names:
        X_plus = X.copy()
        X_minus = X.copy()

        # Apply ±10% change
        X_plus[col] *= (1 + percent_change)
        X_minus[col] *= (1 - percent_change)

        # Predict
        pred_plus = pipeline.predict(X_plus)
        pred_minus = pipeline.predict(X_minus)

        # Sensitivity: average change in prediction
        sensitivity = np.mean(np.abs(pred_plus - base_preds) + np.abs(pred_minus - base_preds)) / 2
        sensitivities.append(sensitivity)

    return pd.DataFrame({
        'Feature': feature_names,
        'Sensitivity': sensitivities
    }).sort_values(by='Sensitivity', ascending=False)

# -------------------------------
# ✅ Run Sensitivity Analysis
# -------------------------------
pipeline.fit(X, y)  # Train full model
sensitivity_df = sensitivity_analysis(pipeline, X.copy(), X.columns.tolist())

# Print results
print("\nFeature Sensitivity:")
print(sensitivity_df)

# Plot Sensitivity
plt.figure(figsize=(10, 6))
sns.barplot(data=sensitivity_df, x='Sensitivity', y='Feature', hue='Feature', palette='rocket', legend=False)
plt.title('Feature Sensitivity Analysis (±10%) - SVR')
plt.xlabel('Average Change in Prediction')
plt.ylabel('Feature')
plt.grid(True)
plt.tight_layout()
plt.show()
