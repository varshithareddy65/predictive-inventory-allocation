
import pandas as pd
import numpy as np

# Load validation predictions
validation = pd.read_parquet(
    "datasets/validation_model_selection.parquet"
)

# Calculate WAPE for each method and combination
summary = (
    validation.groupby(
        ["customer_state", "product_category", "method"]
    )
    .agg(
        total_actual=("actual", "sum"),
        total_absolute_error=("absolute_error", "sum")
    )
    .reset_index()
)

summary["wape_percent"] = (
    summary["total_absolute_error"]
    / summary["total_actual"].replace(0, np.nan)
) * 100

# Select the method with the lowest validation WAPE
selection = (
    summary.sort_values(
        ["customer_state", "product_category",
         "wape_percent", "method"]
    )
    .drop_duplicates(
        ["customer_state", "product_category"]
    )
)

# Keep predictions from the selected method
selected = validation.merge(
    selection[
        ["customer_state", "product_category", "method"]
    ],
    on=[
        "customer_state",
        "product_category",
        "method"
    ],
    how="inner"
)

# Signed error: actual minus predicted
selected["forecast_error"] = (
    selected["actual"] - selected["prediction"]
)

# Estimate uncertainty from validation errors
uncertainty = (
    selected.groupby(
        ["customer_state", "product_category"]
    )
    .agg(
        selected_method=("method", "first"),
        validation_months=("forecast_month", "count"),
        mean_forecast_error=("forecast_error", "mean"),
        error_std=("forecast_error", "std"),
        validation_wape=("absolute_error", "sum")
    )
    .reset_index()
)

# Calculate validation WAPE percentage
actual_totals = (
    selected.groupby(
        ["customer_state", "product_category"]
    )["actual"].sum()
    .reset_index(name="total_actual")
)

uncertainty = uncertainty.merge(
    actual_totals,
    on=["customer_state", "product_category"]
)

uncertainty["validation_wape_percent"] = (
    uncertainty["validation_wape"]
    / uncertainty["total_actual"].replace(0, np.nan)
) * 100

# 90% service-level safety stock factor
z_score = 1.28

uncertainty["safety_stock"] = np.ceil(
    z_score * uncertainty["error_std"].clip(lower=0)
)

uncertainty["safety_stock"] = (
    uncertainty["safety_stock"].fillna(0)
)

# Save results
uncertainty.to_parquet(
    "datasets/validation_safety_stock_inputs.parquet",
    index=False
)

print("\nVALIDATION-BASED SAFETY STOCK INPUTS")
print(
    uncertainty[
        [
            "customer_state",
            "product_category",
            "selected_method",
            "validation_wape_percent",
            "error_std",
            "safety_stock"
        ]
    ].round(2).to_string(index=False)
)

print("\nCombinations:", len(uncertainty))
print("Saved: datasets/validation_safety_stock_inputs.parquet")
