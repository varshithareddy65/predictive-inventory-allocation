
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load datasets
hybrid = pd.read_parquet(
    "datasets/hybrid_forecasting_predictions.parquet"
)

naive = pd.read_parquet(
    "datasets/baseline_forecasting_results.parquet"
)

naive = naive[naive["method"] == "naive"].copy()

# Identify all state-category combinations
combinations = (
    hybrid[["customer_state", "product_category"]]
    .drop_duplicates()
    .sort_values(["customer_state", "product_category"])
    .values
)

n = len(combinations)
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(
    rows, cols,
    figsize=(20, rows * 4),
    sharex=True
)

axes = axes.flatten()

for i, (state, category) in enumerate(combinations):
    ax = axes[i]

    h = hybrid[
        (hybrid["customer_state"] == state)
        & (hybrid["product_category"] == category)
    ].sort_values("forecast_month")

    nv = naive[
        (naive["customer_state"] == state)
        & (naive["product_category"] == category)
    ].sort_values("forecast_month")

    # Actual demand
    ax.plot(
        h["forecast_month"],
        h["actual"],
        marker="o",
        label="Actual"
    )

    # Naive forecast
    ax.plot(
        nv["forecast_month"],
        nv["prediction"],
        marker="o",
        label="Naive"
    )

    # Hybrid forecast
    ax.plot(
        h["forecast_month"],
        h["prediction"],
        marker="o",
        label="Hybrid"
    )

    ax.set_title(f"{state} - {category}", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3)

# Hide unused panels
for j in range(n, len(axes)):
    axes[j].set_visible(False)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1)
)

fig.suptitle(
    "Actual vs Predicted Demand: All 20 State-Category Combinations",
    fontsize=16,
    y=1.02
)

fig.supxlabel("Forecast Month")
fig.supylabel("Item records")

plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])

plt.savefig(
    "datasets/all_forecasts_dashboard.png",
    dpi=250,
    bbox_inches="tight"
)

plt.show()

print("Dashboard created successfully.")
print("Combinations plotted:", n)
print("Saved: datasets/all_forecasts_dashboard.png")
