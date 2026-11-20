
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load August forecast predictions
# --------------------------------------------------

forecasts = pd.read_parquet(
    "datasets/hybrid_forecasting_predictions.parquet"
)

forecasts["forecast_month"] = pd.to_datetime(
    forecasts["forecast_month"]
)

august = forecasts[
    forecasts["forecast_month"] == "2018-08-01"
].copy()

# --------------------------------------------------
# 2. Load validation-based safety stock
# --------------------------------------------------

safety = pd.read_parquet(
    "datasets/validation_safety_stock_inputs.parquet"
)

# --------------------------------------------------
# 3. Merge demand and safety stock
# --------------------------------------------------

allocation = august.merge(
    safety[
        [
            "customer_state",
            "product_category",
            "selected_method",
            "safety_stock"
        ]
    ],
    on=["customer_state", "product_category"],
    how="inner",
    validate="one_to_one"
)

assert len(allocation) == 20

# --------------------------------------------------
# 4. Calculate inventory requirements
# --------------------------------------------------

allocation["forecast_demand"] = (
    allocation["prediction"].clip(lower=0)
)

allocation["required_stock"] = np.ceil(
    allocation["forecast_demand"]
    + allocation["safety_stock"]
).astype(int)

# --------------------------------------------------
# 5. Simulate category-wise available inventory
# --------------------------------------------------

STOCK_AVAILABILITY_RATIO = 0.80

allocation["allocated_stock"] = 0
allocation["available_category_stock"] = 0

for category, group in allocation.groupby("product_category"):

    indices = group.index

    total_requirement = int(
        group["required_stock"].sum()
    )

    available_stock = int(
        np.floor(
            total_requirement * STOCK_AVAILABILITY_RATIO
        )
    )

    # Proportional allocation
    exact_allocations = (
        group["required_stock"]
        / total_requirement
        * available_stock
    )

    base_allocations = np.floor(
        exact_allocations
    ).astype(int)

    # Distribute remaining stock using largest remainders
    remaining = available_stock - int(
        base_allocations.sum()
    )

    fractions = (
        exact_allocations - base_allocations
    )

    order = fractions.sort_values(
        ascending=False,
        kind="stable"
    ).index

    final_allocations = base_allocations.copy()

    for idx in order[:remaining]:
        final_allocations.loc[idx] += 1

    allocation.loc[
        indices, "allocated_stock"
    ] = final_allocations

    allocation.loc[
        indices, "available_category_stock"
    ] = available_stock

# --------------------------------------------------
# 6. Calculate shortage and coverage
# --------------------------------------------------

# 6. Calculate shortage and coverage

allocation["shortage"] = (
    allocation["required_stock"]
    - allocation["allocated_stock"]
)

allocation["requirement_coverage_percent"] = (
    allocation["allocated_stock"]
    / allocation["required_stock"]
    * 100
)

allocation["forecast_demand_coverage_percent"] = (
    allocation["allocated_stock"]
    / allocation["forecast_demand"].replace(0, np.nan)
    * 100
).fillna(0)

allocation["unmet_forecast_demand"] = (
    allocation["forecast_demand"]
    - allocation["allocated_stock"]
).clip(lower=0)

allocation["remaining_safety_stock"] = (
    allocation["allocated_stock"]
    - allocation["forecast_demand"]
).clip(lower=0)

# --------------------------------------------------
# 7. Validate inventory constraints
# --------------------------------------------------

assert (
    allocation["allocated_stock"]
    <= allocation["required_stock"]
).all()

assert (
    allocation["allocated_stock"] >= 0
).all()

category_check = allocation.groupby(
    "product_category"
).agg(
    available_stock=("available_category_stock", "first"),
    allocated_stock=("allocated_stock", "sum"),
    required_stock=("required_stock", "sum")
)

assert (
    category_check["available_stock"]
    == category_check["allocated_stock"]
).all()

# --------------------------------------------------
# 8. Save allocation recommendations
# --------------------------------------------------

allocation.to_parquet(
    "datasets/inventory_allocation_recommendations.parquet",
    index=False
)

allocation.to_csv(
    "datasets/inventory_allocation_recommendations.csv",
    index=False
)

# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\nINVENTORY ALLOCATION RECOMMENDATIONS")


print(
    allocation[
        [
            "customer_state",
            "product_category",
            "forecast_demand",
            "safety_stock",
            "required_stock",
            "allocated_stock",
            "shortage",
            "requirement_coverage_percent",
            "forecast_demand_coverage_percent",
            "unmet_forecast_demand",
            "remaining_safety_stock"
        ]
    ].round(2).to_string(index=False)
)


print("\nCATEGORY-WISE INVENTORY SUMMARY")

print(
    category_check.to_string()
)

print("\nOVERALL SUMMARY")

print("Total required stock:",
      allocation["required_stock"].sum())

print("Total allocated stock:",
      allocation["allocated_stock"].sum())

print("Total shortage:",
      allocation["shortage"].sum())

print(
    "Requirement coverage:",
    round(
        allocation["allocated_stock"].sum()
        / allocation["required_stock"].sum() * 100,
        2
    ),
    "%"
)

print(
    "Forecast demand coverage:",
    round(
        allocation["allocated_stock"].sum()
        / allocation["forecast_demand"].sum() * 100,
        2
    ),
    "%"
)

print(
    "Unmet forecast demand:",
    round(allocation["unmet_forecast_demand"].sum(), 2)
)

print(
    "Remaining safety stock:",
    round(allocation["remaining_safety_stock"].sum(), 2)
)

print("\nSaved allocation recommendations successfully.")
