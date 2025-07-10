import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
file_path = 'final_isotropic.csv'  # Replace with your actual file
df = pd.read_csv(file_path)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Features and target
X = df.drop(['Flexure_SR'], axis=1)
y = df['Flexure_SR']
feature_names = X.columns.tolist()

# Define pipeline: Scaling + MLP
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))
])

# Define K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated R² and MSE scores
r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
mse_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')

print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")

# Get cross-validated predictions
y_pred = cross_val_predict(pipeline, X, y, cv=kf)
overall_r2 = r2_score(y, y_pred)

# Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, color='blueviolet', label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
plt.title(f'MLP: Actual vs Predicted\nR² = {overall_r2:.4f}')
plt.xlabel('Actual Flexure_SR')
plt.ylabel('Predicted Flexure_SR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot R² and MSE scores for each fold
fold_labels = [f'Fold {i}' for i in range(1, len(r2_scores) + 1)]
plt.figure(figsize=(14, 6))

# R² plot
plt.subplot(1, 2, 1)
sns.barplot(x=fold_labels, y=r2_scores, palette='cool', legend=False)
plt.axhline(np.mean(r2_scores), color='red', linestyle='--', label=f'Mean R² = {np.mean(r2_scores):.4f}')
plt.title('MLP: R² Score per Fold')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.ylim(-1, 1)
plt.legend()
plt.grid(True)

# MSE plot
plt.subplot(1, 2, 2)
sns.barplot(x=fold_labels, y=mse_scores, palette='cividis', legend=False)
plt.axhline(np.mean(mse_scores), color='red', linestyle='--', label=f'Mean MSE = {np.mean(mse_scores):.4f}')
plt.title('MLP: MSE per Fold')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# -------------------------------
# ✅ One-at-a-Time Sensitivity Analysis
# -------------------------------
def one_feature_sensitivity(pipeline, X, feature_names, percent_change=0.1):
    base_preds = pipeline.predict(X)
    sensitivities = []

    for col in feature_names:
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[col] = X[col] * (1 + percent_change)
        X_minus[col] = X[col] * (1 - percent_change)

        pred_plus = pipeline.predict(X_plus)
        pred_minus = pipeline.predict(X_minus)

        sensitivity = np.mean(np.abs(pred_plus - base_preds) + np.abs(pred_minus - base_preds)) / 2
        sensitivities.append(sensitivity)

    return pd.DataFrame({
        'Feature': feature_names,
        'Sensitivity': sensitivities
    }).sort_values(by='Sensitivity', ascending=False)

# Train model and run sensitivity
pipeline.fit(X, y)
sensitivity_df = one_feature_sensitivity(pipeline, X.copy(), feature_names)

print("\nFeature Sensitivity (One-at-a-Time ±10%):")
print(sensitivity_df)

# Plot Sensitivity
plt.figure(figsize=(10, 6))
sns.barplot(data=sensitivity_df, x='Sensitivity', y='Feature', hue='Feature', palette='plasma', legend=False)
plt.title('MLP: Sensitivity Analysis (±10%) — One Feature at a Time')
plt.xlabel('Avg Change in Prediction')
plt.ylabel('Feature')
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------------
# ✅ Linear Sensitivity Plots in Grid
# -------------------------------
def plot_linear_sensitivity_grid(pipeline, X, feature_names, percent_range=0.1, steps=21):
    X_mean = X.mean().to_frame().T
    percent_values = np.linspace(1 - percent_range, 1 + percent_range, steps)

    n_features = len(feature_names)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_names):
        X_test = pd.concat([X_mean] * steps, ignore_index=True)
        X_test[col] = X[col].mean() * percent_values
        y_pred = pipeline.predict(X_test)

        axes[i].plot(X_test[col], y_pred, marker='o', color='darkorange')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Predicted Flexure_SR')
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('MLP: Linear Sensitivity (±10%) — One Feature at a Time', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Plot linear sensitivity
plot_linear_sensitivity_grid(pipeline, X, feature_names)
