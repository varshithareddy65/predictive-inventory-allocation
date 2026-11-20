
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = "datasets/forecasting_experiment_top20.parquet"
OUTPUT = "datasets/baseline_forecasting_results.parquet"

Path("datasets/duckdb_tmp").mkdir(
    parents=True, exist_ok=True
)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='datasets/duckdb_tmp'")

df = con.execute(f"""
    SELECT
        order_month,
        customer_state,
        product_category,
        demand
    FROM read_parquet('{INPUT}')
    ORDER BY customer_state, product_category, order_month
""").fetchdf()

con.close()

df["order_month"] = pd.to_datetime(df["order_month"])

results = []

for (state, category), group in df.groupby(
    ["customer_state", "product_category"]
):
    group = group.sort_values("order_month").reset_index(drop=True)

    # First 15 months are initial history.
    # Final 8 months are evaluated sequentially.
    for i in range(15, len(group)):
        history = group.loc[:i - 1, "demand"].to_numpy()
        actual = float(group.loc[i, "demand"])

        naive = float(history[-1])
        moving_average = float(np.mean(history[-3:]))

        # Use same month in previous year only when available.
        seasonal_naive = (
            float(history[-12])
            if len(history) >= 12
            else np.nan
        )

        for method, prediction in [
            ("naive", naive),
            ("moving_average_3", moving_average),
            ("seasonal_naive_12", seasonal_naive),
        ]:
            if not np.isnan(prediction):
                results.append({
                    "customer_state": state,
                    "product_category": category,
                    "forecast_month": group.loc[i, "order_month"],
                    "method": method,
                    "actual": actual,
                    "prediction": max(0.0, prediction)
                })

results_df = pd.DataFrame(results)

results_df.to_parquet(
    OUTPUT,
    index=False,
    compression="zstd"
)

results_df["absolute_error"] = (
    results_df["actual"] - results_df["prediction"]
).abs()

results_df["squared_error"] = (
    results_df["actual"] - results_df["prediction"]
) ** 2

print("\nFORECASTING BASELINE EVALUATION")

summary = results_df.groupby("method").agg(
    forecasts=("actual", "count"),
    MAE=("absolute_error", "mean"),
    RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
    total_actual=("actual", "sum"),
    total_absolute_error=("absolute_error", "sum")
).reset_index()

summary["WAPE_percent"] = (
    summary["total_absolute_error"] /
    summary["total_actual"].replace(0, np.nan)
) * 100

print(summary.to_string(index=False))

print("\nFORECAST COUNTS BY METHOD")
print(results_df.groupby("method").size().to_string())

print("\nSaved:", OUTPUT)
