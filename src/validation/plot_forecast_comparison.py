
import pandas as pd
import matplotlib.pyplot as plt

hybrid = pd.read_parquet(
    "datasets/hybrid_forecasting_predictions.parquet"
)

naive = pd.read_parquet(
    "datasets/baseline_forecasting_results.parquet"
)

holt = pd.read_parquet(
    "datasets/holt_forecasting_predictions.parquet"
)

naive = naive[naive["method"] == "naive"]

state = "SP"
category = "toys"

def get_series(df, name):
    data = df[
        (df["customer_state"] == state)
        & (df["product_category"] == category)
    ].copy()

    data = data.sort_values("forecast_month")

    return data.set_index("forecast_month")["prediction"].rename(name)

actual = hybrid[
    (hybrid["customer_state"] == state)
    & (hybrid["product_category"] == category)
].sort_values("forecast_month").set_index("forecast_month")["actual"]

comparison = pd.concat(
    [
        actual.rename("Actual"),
        get_series(naive, "Naive"),
        get_series(holt, "Holt"),
        get_series(hybrid, "Hybrid")
    ],
    axis=1
)

comparison.plot(
    figsize=(12, 6),
    marker="o"
)

plt.title("SP - Bed Bath Table: Actual vs Forecast")
plt.xlabel("Forecast Month")
plt.ylabel("Item records")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    "datasets/sp_bed_bath_table_forecast.png",
    dpi=300
)

plt.show()

print(comparison.round(2))
print("\nSaved forecast graph.")
