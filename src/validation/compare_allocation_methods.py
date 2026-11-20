
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets"

V1_FILE = DATA_DIR / "inventory_allocation_recommendations.parquet"
V2_FILE = DATA_DIR / "inventory_allocation_recommendations_v2.parquet"

OUTPUT_DIR = DATA_DIR


# --------------------------------------------------
# 2. LOAD RESULTS
# --------------------------------------------------

print("\nLoading V1 and V2 allocation results...")

v1 = pd.read_parquet(V1_FILE)
v2 = pd.read_parquet(V2_FILE)

v1["method"] = "V1"
v2["method"] = "V2"

print("V1 rows:", len(v1))
print("V2 rows:", len(v2))

print("\nColumns:")
print(v1.columns.tolist())


# --------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "customer_state",
    "product_category",
    "required_stock",
    "allocated_stock",
    "shortage",
    "unmet_forecast_demand",
    "remaining_safety_stock",
]

for column in required_columns:
    if column not in v1.columns or column not in v2.columns:
        raise ValueError(f"Required column missing: {column}")


# --------------------------------------------------
# 4. OVERALL COMPARISON
# --------------------------------------------------

metrics = [
    "required_stock",
    "allocated_stock",
    "shortage",
    "unmet_forecast_demand",
    "remaining_safety_stock",
]

overall = pd.DataFrame({
    "Metric": metrics,
    "V1": [v1[column].sum() for column in metrics],
    "V2": [v2[column].sum() for column in metrics],
})

overall["Difference_V2_minus_V1"] = overall["V2"] - overall["V1"]

print("\n========== OVERALL COMPARISON ==========")
print(overall.to_string(index=False))

overall.to_csv(
    OUTPUT_DIR / "allocation_overall_comparison.csv",
    index=False
)


# --------------------------------------------------
# 5. UNMET DEMAND IMPROVEMENT
# --------------------------------------------------

v1_unmet = v1["unmet_forecast_demand"].sum()
v2_unmet = v2["unmet_forecast_demand"].sum()

if v1_unmet > 0:
    improvement = ((v1_unmet - v2_unmet) / v1_unmet) * 100
else:
    improvement = 0

print("\n========== DEMAND COVERAGE ==========")
print(f"V1 unmet forecast demand: {v1_unmet:.2f}")
print(f"V2 unmet forecast demand: {v2_unmet:.2f}")
print(f"Reduction in unmet demand: {improvement:.2f}%")


# --------------------------------------------------
# 6. CATEGORY-WISE COMPARISON
# --------------------------------------------------

category_metrics = [
    "required_stock",
    "allocated_stock",
    "shortage",
    "unmet_forecast_demand",
    "remaining_safety_stock",
]

v1_category = (
    v1.groupby("product_category")[category_metrics]
    .sum()
)

v2_category = (
    v2.groupby("product_category")[category_metrics]
    .sum()
)

category_comparison = v1_category.add_suffix("_V1").join(
    v2_category.add_suffix("_V2")
)

category_comparison["unmet_demand_reduction"] = (
    category_comparison["unmet_forecast_demand_V1"]
    - category_comparison["unmet_forecast_demand_V2"]
)

category_comparison = category_comparison.reset_index()

category_comparison.to_csv(
    OUTPUT_DIR / "allocation_category_comparison.csv",
    index=False
)

print("\n========== CATEGORY COMPARISON ==========")
print(category_comparison.to_string(index=False))


# --------------------------------------------------
# 7. STATE-CATEGORY COMPARISON
# --------------------------------------------------

state_category = v1[
    ["customer_state", "product_category"] + category_metrics
].merge(
    v2[
        ["customer_state", "product_category"] + category_metrics
    ],
    on=["customer_state", "product_category"],
    suffixes=("_V1", "_V2"),
    validate="one_to_one",
)

state_category["unmet_demand_reduction"] = (
    state_category["unmet_forecast_demand_V1"]
    - state_category["unmet_forecast_demand_V2"]
)

state_category.to_csv(
    OUTPUT_DIR / "allocation_state_category_comparison.csv",
    index=False
)


# --------------------------------------------------
# 8. ALLOCATION COMPARISON CHART
# --------------------------------------------------

chart_metrics = [
    "allocated_stock",
    "unmet_forecast_demand",
    "remaining_safety_stock",
]

chart_data = pd.DataFrame({
    "V1": [v1[column].sum() for column in chart_metrics],
    "V2": [v2[column].sum() for column in chart_metrics],
}, index=chart_metrics)

ax = chart_data.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.7
)

ax.set_title("Inventory Allocation: V1 vs V2")
ax.set_xlabel("Performance Metric")
ax.set_ylabel("Stock / Demand Records")
ax.legend(title="Allocation Method")
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "allocation_v1_v2_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 9. UNMET DEMAND BY CATEGORY CHART
# --------------------------------------------------

category_plot = category_comparison.set_index("product_category")[
    [
        "unmet_forecast_demand_V1",
        "unmet_forecast_demand_V2",
    ]
]

ax = category_plot.plot(
    kind="bar",
    figsize=(13, 6),
    width=0.8
)

ax.set_title("Unmet Forecast Demand by Category")
ax.set_xlabel("Product Category")
ax.set_ylabel("Unmet Forecast Demand")
ax.legend(["V1", "V2"])
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "unmet_demand_by_category.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n========== COMPARISON COMPLETED ==========")
print(f"Unmet demand improvement: {improvement:.2f}%")

print("\nGenerated files:")
print("1. allocation_overall_comparison.csv")
print("2. allocation_category_comparison.csv")
print("3. allocation_state_category_comparison.csv")
print("4. allocation_v1_v2_comparison.png")
print("5. unmet_demand_by_category.png")

print("\nV1 vs V2 comparison completed successfully!")
