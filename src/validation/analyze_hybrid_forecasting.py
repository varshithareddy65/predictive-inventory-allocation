
import pandas as pd
import numpy as np

# Load predictions
hybrid = pd.read_parquet(
    "datasets/hybrid_forecasting_predictions.parquet"
)

naive = pd.read_parquet(
    "datasets/baseline_forecasting_results.parquet"
)

holt = pd.read_parquet(
    "datasets/holt_forecasting_predictions.parquet"
)

naive = naive[naive["method"] == "naive"].copy()

def calculate_metrics(df, model_name):
    result = (
        df.groupby(["customer_state", "product_category"])
        .agg(
            total_actual=("actual", "sum"),
            total_absolute_error=("absolute_error", "sum")
        )
        .reset_index()
    )

    result[f"{model_name}_wape"] = (
        result["total_absolute_error"]
        / result["total_actual"].replace(0, np.nan)
    ) * 100

    return result[
        ["customer_state", "product_category",
         f"{model_name}_wape"]
    ]

# Calculate absolute errors
for df in [hybrid, naive, holt]:
    df["absolute_error"] = (
        df["actual"] - df["prediction"]
    ).abs()

hybrid_metrics = calculate_metrics(hybrid, "hybrid")
naive_metrics = calculate_metrics(naive, "naive")
holt_metrics = calculate_metrics(holt, "holt")

# Combine model comparisons
comparison = hybrid_metrics.merge(
    naive_metrics,
    on=["customer_state", "product_category"]
).merge(
    holt_metrics,
    on=["customer_state", "product_category"]
)

comparison["best_model"] = comparison[
    ["hybrid_wape", "naive_wape", "holt_wape"]
].idxmin(axis=1).str.replace("_wape", "")

comparison["hybrid_vs_naive"] = (
    comparison["hybrid_wape"] - comparison["naive_wape"]
)

comparison = comparison.sort_values("hybrid_wape")

print("\nPER-COMBINATION FORECASTING COMPARISON")
print(comparison.round(2).to_string(index=False))

print("\nBEST MODEL COUNTS ON TEST DATA")
print(comparison["best_model"].value_counts().to_string())

print("\nHYBRID IMPROVEMENTS OVER NAIVE")
print(
    (comparison["hybrid_vs_naive"] < 0).sum(),
    "out of",
    len(comparison),
    "combinations"
)

comparison.to_csv(
    "datasets/hybrid_performance_comparison.csv",
    index=False
)

print("\nSaved: datasets/hybrid_performance_comparison.csv")

