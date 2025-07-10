import pandas as pd
import numpy as np

def augment_one_column_at_a_time(
    input_csv_path,
    output_csv_path,
    variation_percent=10,
    target_column=None,
    random_state=None
):
    """
    Augments one column at a time (±variation_percent) and saves combined output to CSV.

    Parameters:
    - input_csv_path: str, path to the input CSV file
    - output_csv_path: str, path to the output CSV file
    - variation_percent: float, percentage of noise (e.g., 10 for ±10%)
    - target_column: str or None, column to exclude from augmentation (e.g., output variable)
    - random_state: int or None, for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)

    df = pd.read_csv(input_csv_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    augmented_dfs = [df]  # start with the original

    for col in numeric_cols:
        df_aug = df.copy()
        variation = (variation_percent / 100.0) * df[col]
        noise = np.random.uniform(-variation, variation)
        df_aug[col] = df[col] + noise

        augmented_dfs.append(df_aug)

    final_df = pd.concat(augmented_dfs, ignore_index=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"Augmented data (1 column at a time) saved to: {output_csv_path}")

# ---------------- Example Usage ----------------

input_file = "final_isotropic.csv"
output_file = "final_isotropic_augmented.csv"
augment_one_column_at_a_time(
    input_csv_path=input_file,
    output_csv_path=output_file,
    variation_percent=10,
    target_column="Flexure_SR",  # skip target column
    random_state=42
)
