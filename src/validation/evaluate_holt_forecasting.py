
import duckdb
import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing

INPUT = "datasets/forecasting_experiment_top20.parquet"
OUTPUT = "datasets/holt_forecasting_predictions.parquet"

con = duckdb.connect()
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
    # Forecast January-August 2018 sequentially.
    for i in range(15, len(group)):
        history = group.loc[:i - 1, "demand"].astype(float).to_numpy()
        actual = float(group.loc[i, "demand"])

        try:
            model = ExponentialSmoothing(
                history,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated"
            )

            fitted = model.fit(optimized=True)
            prediction = float(fitted.forecast(1)[0])
            prediction = max(0.0, prediction)

        except Exception as error:
            print(
                f"Skipped {state} {category} "
                f"{group.loc[i, 'order_month']}: {error}"
            )
            continue

        results.append({
            "customer_state": state,
            "product_category": category,
            "forecast_month": group.loc[i, "order_month"],
            "actual": actual,
            "prediction": prediction
        })

results_df = pd.DataFrame(results)

results_df["absolute_error"] = (
    results_df["actual"] - results_df["prediction"]
).abs()

results_df["squared_error"] = (
    results_df["actual"] - results_df["prediction"]
) ** 2

mae = results_df["absolute_error"].mean()
rmse = np.sqrt(results_df["squared_error"].mean())

total_actual = results_df["actual"].sum()
wape = (
    results_df["absolute_error"].sum()
    / total_actual * 100
)

results_df.to_parquet(
    OUTPUT,
    index=False,
    compression="zstd"
)

print("\nHOLT LINEAR TREND RESULTS")
print("Forecasts:", len(results_df))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("WAPE:", round(wape, 4), "%")

print("\nSaved:", OUTPUT)
