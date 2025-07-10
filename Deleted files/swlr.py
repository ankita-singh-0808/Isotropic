import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'final_isotropic.csv'  # Replace with actual file path
df = pd.read_csv(file_path)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['LoadPosition'], drop_first=True)

# Feature and target separation
X = df.drop(['Flexure_SR'], axis=1)
y = df['Flexure_SR']

# Convert all data to numeric to avoid dtype=object issues
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop any rows with NaN (can happen after coercion)
valid_indices = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_indices]
y = y.loc[valid_indices]

# Stepwise regression function
def stepwise_selection(X, y, 
                       initial_features=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    included = list(initial_features)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            try:
                model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            except Exception as e:
                if verbose:
                    print(f"Skipping {new_column} due to error: {e}")
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"Add  {best_feature:30} with p-value {best_pval:.6f}")

        # backward step
        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.iloc[1:]  # exclude intercept
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f"Drop {worst_feature:30} with p-value {worst_pval:.6f}")
        if not changed:
            break
    return included

# Feature selection
selected_features = stepwise_selection(X, y)
print("Selected features:", selected_features)

# Final model with selected features
X_selected = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Fit model
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# Predict
y_train_pred = model.predict(sm.add_constant(X_train))
y_test_pred = model.predict(sm.add_constant(X_test))

# Metrics
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"\nTraining MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
print(f"Testing MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

axs[0].scatter(y_train, y_train_pred, alpha=0.7, color='green')
axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
axs[0].set_title(f'Training: Actual vs Predicted\nR² = {r2_train:.4f}')
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')
axs[0].grid(True)

axs[1].scatter(y_test, y_test_pred, alpha=0.7, color='blue')
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axs[1].set_title(f'Testing: Actual vs Predicted\nR² = {r2_test:.4f}')
axs[1].set_xlabel('Actual')
axs[1].set_ylabel('Predicted')
axs[1].grid(True)

plt.tight_layout()
plt.show()
