import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Load & preprocess data
# -------------------------
df = pd.read_csv('final_isotropic_augmented.csv')
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

X = df.drop(['Flexure_SR'], axis=1)
y = df['Flexure_SR']
feature_names = X.columns.tolist()

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Pipeline: scaling + SVR
# -------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# -------------------------
# R² scores
# -------------------------
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train R²: {r2_train:.4f}")
print(f"Test R²:  {r2_test:.4f}")

# -------------------------
# Plot: Actual vs Predicted
# -------------------------
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

axs[0].scatter(y_train, y_train_pred, alpha=0.6, color='green', label='Training')
axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
axs[0].set_title(f'Training: Actual vs Predicted\nR² = {r2_train:.4f}')
axs[0].set_xlabel('Actual Flexure_SR')
axs[0].set_ylabel('Predicted Flexure_SR')
axs[0].legend()
axs[0].grid(True)

axs[1].scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Testing')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axs[1].set_title(f'Testing: Actual vs Predicted\nR² = {r2_test:.4f}')
axs[1].set_xlabel('Actual Flexure_SR')
axs[1].set_ylabel('Predicted Flexure_SR')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# -------------------------
# K-Fold CV: Train & Test R²
# -------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_r2_scores = []
test_r2_scores = []

for train_idx, test_idx in kf.split(X):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train_fold, y_train_fold)

    y_train_pred = pipeline.predict(X_train_fold)
    y_test_pred = pipeline.predict(X_test_fold)

    r2_train = r2_score(y_train_fold, y_train_pred)
    r2_test = r2_score(y_test_fold, y_test_pred)

    train_r2_scores.append(r2_train)
    test_r2_scores.append(r2_test)

# Report
print(f"\n✅ Mean Train R²: {np.mean(train_r2_scores):.4f} ± {np.std(train_r2_scores):.4f}")
print(f"✅ Mean Test  R²: {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")

# -------------------------
# Sensitivity Analysis Plot (2-cols)
# -------------------------
def plot_linear_sensitivity_grid(pipeline, X, feature_names, steps=50):
    X_mean = X.mean().to_frame().T

    # Remove dummy features like LoadPosition_EL, IL
    filtered_features = [col for col in feature_names if not col.startswith("LoadPosition_")]

    n_features = len(filtered_features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(filtered_features):
        X_test = pd.concat([X_mean] * steps, ignore_index=True)

        if col.lower() == 'thickness':
            X_test[col] = np.linspace(60, 150, steps)
        else:
            X_test[col] = np.linspace(X[col].min(), X[col].max(), steps)

        y_pred = pipeline.predict(X_test)

        axes[i].plot(X_test[col], y_pred, marker='o', color='teal')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Predicted Flexure_SR')
        axes[i].grid(True)

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('SVR: Sensitivity Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Call function
plot_linear_sensitivity_grid(pipeline, X, feature_names)
