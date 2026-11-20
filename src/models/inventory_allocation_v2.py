
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Load forecast and safety-stock data
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

safety = pd.read_parquet(
    "datasets/validation_safety_stock_inputs.parquet"
)

allocation = august.merge(
    safety[
        [
            "customer_state",
            "product_category",
            "safety_stock",
            "selected_method"
        ]
    ],
    on=["customer_state", "product_category"],
    how="inner",
    validate="one_to_one"
)

assert len(allocation) == 20


# --------------------------------------------------
# 2. Calculate inventory requirements
# --------------------------------------------------

allocation["forecast_demand"] = (
    allocation["prediction"].clip(lower=0)
)

allocation["demand_target"] = np.ceil(
    allocation["forecast_demand"]
).astype(int)

allocation["safety_stock"] = (
    allocation["safety_stock"].clip(lower=0)
)

allocation["required_stock"] = np.ceil(
    allocation["forecast_demand"]
    + allocation["safety_stock"]
).astype(int)

allocation["allocated_stock"] = 0
allocation["available_category_stock"] = 0


# --------------------------------------------------
# 3. Proportional allocation with capacity limits
# --------------------------------------------------

def allocate_proportionally(weights, total, capacities):
    weights = weights.astype(float)
    capacities = capacities.astype(int)

    result = pd.Series(0, index=weights.index, dtype=int)
    remaining = int(total)

    while remaining > 0:
        active = capacities[capacities > result]

        if active.empty:
            break

        active_weights = weights.loc[active.index]

        if active_weights.sum() <= 0:
            active_weights = (
                capacities.loc[active.index]
                - result.loc[active.index]
            ).astype(float)

        quotas = (
            remaining
            * active_weights
            / active_weights.sum()
        )

        floors = np.floor(quotas).astype(int)

        additions = pd.Series(
            np.minimum(
                floors.values,
                (capacities.loc[active.index]
                 - result.loc[active.index]).values
            ),
            index=active.index
        )

        result.loc[active.index] += additions
        remaining -= int(additions.sum())

        if remaining <= 0:
            break

        active = capacities[capacities > result]

        if active.empty:
            break

        fractions = quotas - floors
        order = fractions.loc[active.index].sort_values(
            ascending=False,
            kind="stable"
        ).index

        distributed = 0

        for idx in order:
            if remaining == 0:
                break

            if result.loc[idx] < capacities.loc[idx]:
                result.loc[idx] += 1
                remaining -= 1
                distributed += 1

        if additions.sum() == 0 and distributed == 0:
            break

    return result


# --------------------------------------------------
# 4. Simulate category-wise inventory availability
# --------------------------------------------------

STOCK_AVAILABILITY_RATIO = 0.80

for category, group in allocation.groupby("product_category"):

    indices = group.index

    total_requirement = int(
        group["required_stock"].sum()
    )

    available_stock = int(
        np.floor(
            total_requirement
            * STOCK_AVAILABILITY_RATIO
        )
    )

    allocation.loc[
        indices, "available_category_stock"
    ] = available_stock

    # ----------------------------------------------
    # Stage 1: Allocate towards forecast demand
    # ----------------------------------------------

    demand_capacity = group["demand_target"]

    demand_total = min(
        available_stock,
        int(demand_capacity.sum())
    )

    demand_allocations = allocate_proportionally(
        weights=group["forecast_demand"],
        total=demand_total,
        capacities=demand_capacity
    )

    # ----------------------------------------------
    # Stage 2: Allocate remaining stock to safety
    # ----------------------------------------------

    remaining_stock = (
        available_stock
        - int(demand_allocations.sum())
    )

    safety_capacity = (
        group["required_stock"]
        - demand_allocations
    )

    safety_weights = group["safety_stock"]

    safety_allocations = allocate_proportionally(
        weights=safety_weights,
        total=remaining_stock,
        capacities=safety_capacity
    )

    final_allocations = (
        demand_allocations + safety_allocations
    )

    allocation.loc[
        indices, "allocated_stock"
    ] = final_allocations


# --------------------------------------------------
# 5. Calculate performance metrics
# --------------------------------------------------

allocation["shortage"] = (
    allocation["required_stock"]
    - allocation["allocated_stock"]
)

allocation["requirement_coverage_percent"] = (
    allocation["allocated_stock"]
    / allocation["required_stock"]
    * 100
)

allocation["unmet_forecast_demand"] = (
    allocation["forecast_demand"]
    - allocation["allocated_stock"]
).clip(lower=0)

allocation["remaining_safety_stock"] = (
    allocation["allocated_stock"]
    - allocation["forecast_demand"]
).clip(lower=0)


# --------------------------------------------------
# 6. Validate allocation constraints
# --------------------------------------------------

assert (
    allocation["allocated_stock"] >= 0
).all()

assert (
    allocation["allocated_stock"]
    <= allocation["required_stock"]
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
# 7. Save V2 recommendations
# --------------------------------------------------

allocation.to_parquet(
    "datasets/inventory_allocation_recommendations_v2.parquet",
    index=False
)

allocation.to_csv(
    "datasets/inventory_allocation_recommendations_v2.csv",
    index=False
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nINVENTORY ALLOCATION V2")

print(
    allocation[
        [
            "customer_state",
            "product_category",
            "forecast_demand",
            "safety_stock",
            "required_stock",
            "available_category_stock",
            "allocated_stock",
            "shortage",
            "unmet_forecast_demand",
            "remaining_safety_stock"
        ]
    ].round(2).to_string(index=False)
)

print("\nCATEGORY-WISE SUMMARY")

print(category_check.to_string())

print("\nOVERALL SUMMARY")

print("Required stock:",
      allocation["required_stock"].sum())

print(
    "Available stock:",
    category_check["available_stock"].sum()
)

print("Allocated stock:",
      allocation["allocated_stock"].sum())

print("Total requirement shortage:",
      allocation["shortage"].sum())

print("Unmet forecast demand:",
      round(allocation["unmet_forecast_demand"].sum(), 2))

print("Remaining safety stock:",
      round(allocation["remaining_safety_stock"].sum(), 2))

print("\nSaved V2 allocation recommendations.")
