
import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect()

# Load validation predictions
validation = con.execute("""
    SELECT *
    FROM read_parquet(
        'datasets/validation_model_selection.parquet'
    )
""").fetchdf()

# Calculate validation WAPE for each method and combination
summary = validation.groupby(
    ["customer_state", "product_category", "method"]
).agg(
    total_actual=("actual", "sum"),
    total_absolute_error=(
        "absolute_error",
        "sum"
    )
).reset_index()

summary["wape_percent"] = (
    summary["total_absolute_error"]
    / summary["total_actual"].replace(0, np.nan)
) * 100

# Select the best method using validation only
selection = (
    summary.sort_values(
        ["customer_state", "product_category",
         "wape_percent", "method"]
    )
    .drop_duplicates(
        ["customer_state", "product_category"]
    )
)

# Load final test predictions
naive = con.execute("""
    SELECT customer_state, product_category,
           forecast_month, actual, prediction
    FROM read_parquet(
        'datasets/baseline_forecasting_results.parquet'
    )
    WHERE method = 'naive'
""").fetchdf()

holt = con.execute("""
    SELECT customer_state, product_category,
           forecast_month, actual, prediction
    FROM read_parquet(
        'datasets/holt_forecasting_predictions.parquet'
    )
""").fetchdf()

con.close()

naive["method"] = "naive"
holt["method"] = "holt"

candidates = pd.concat(
    [naive, holt],
    ignore_index=True
)

hybrid = candidates.merge(
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

# Verify identical test coverage
assert len(hybrid) == 160
assert hybrid.groupby(
    ["customer_state", "product_category"]
).size().eq(8).all()

hybrid["absolute_error"] = (
    hybrid["actual"] - hybrid["prediction"]
).abs()

hybrid["squared_error"] = (
    hybrid["actual"] - hybrid["prediction"]
) ** 2

mae = hybrid["absolute_error"].mean()
rmse = np.sqrt(hybrid["squared_error"].mean())

wape = (
    hybrid["absolute_error"].sum()
    / hybrid["actual"].sum()
) * 100

print("\nHYBRID FORECASTING RESULTS")
print("Forecasts:", len(hybrid))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("WAPE:", round(wape, 4), "%")

print("\nSELECTED METHOD COUNTS")
print(selection["method"].value_counts().to_string())

hybrid.to_parquet(
    "datasets/hybrid_forecasting_predictions.parquet",
    index=False,
    compression="zstd"
)

print("\nSaved: datasets/hybrid_forecasting_predictions.parquet")
