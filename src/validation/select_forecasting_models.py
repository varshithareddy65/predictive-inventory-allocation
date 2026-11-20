
import duckdb
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path

INPUT = "datasets/forecasting_experiment_top20.parquet"
OUTPUT = "datasets/validation_model_selection.parquet"

con = duckdb.connect()
df = con.execute(f"""
    SELECT order_month, customer_state, product_category, demand
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

    # April-December 2017: validation
    for i in range(6, 15):
        history = group.loc[:i - 1, "demand"].astype(float).to_numpy()
        actual = float(group.loc[i, "demand"])

        naive_prediction = max(0.0, float(history[-1]))

        try:
            holt_model = ExponentialSmoothing(
                history,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated"
            )
            fitted = holt_model.fit(optimized=True)
            holt_prediction = max(
                0.0, float(fitted.forecast(1)[0])
            )

        except Exception as error:
            print(f"Holt failed for {state}, {category}: {error}")
            continue

        for method, prediction in [
            ("naive", naive_prediction),
            ("holt", holt_prediction)
        ]:
            results.append({
                "customer_state": state,
                "product_category": category,
                "forecast_month": group.loc[i, "order_month"],
                "method": method,
                "actual": actual,
                "prediction": prediction
            })

results_df = pd.DataFrame(results)

results_df["absolute_error"] = (
    results_df["actual"] - results_df["prediction"]
).abs()

summary = results_df.groupby(
    ["customer_state", "product_category", "method"]
).agg(
    forecasts=("actual", "count"),
    total_actual=("actual", "sum"),
    total_absolute_error=("absolute_error", "sum")
).reset_index()

summary["wape_percent"] = (
    summary["total_absolute_error"]
    / summary["total_actual"].replace(0, np.nan)
) * 100

# Select the method with the lower validation WAPE.
# If tied, choose naive.
selection = (
    summary.sort_values(
        ["customer_state", "product_category", "wape_percent", "method"],
        ascending=[True, True, True, True]
    )
    .drop_duplicates(["customer_state", "product_category"])
)

results_df.to_parquet(
    OUTPUT,
    index=False,
    compression="zstd"
)

print("\nVALIDATION SUMMARY")

print(summary.to_string(index=False))

print("\nMODEL SELECTION COUNTS")
print(selection["method"].value_counts().to_string())

print("\nSELECTED METHODS")
print(selection[
    ["customer_state", "product_category", "method", "wape_percent"]
].to_string(index=False))

print("\nSaved:", OUTPUT)
