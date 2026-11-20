
import duckdb
from pathlib import Path

INPUT = "datasets/olist_state_category_status_monthly.parquet"
OUTPUT = "datasets/forecasting_experiment_top20.parquet"

Path("datasets/duckdb_tmp").mkdir(
    parents=True, exist_ok=True
)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='datasets/duckdb_tmp'")

query = f"""
COPY (
    WITH history AS (
        SELECT *
        FROM read_parquet('{INPUT}')
        WHERE order_month BETWEEN
            DATE '2016-10-01' AND DATE '2018-08-01'
    ),

    top_combinations AS (
        SELECT
            customer_state,
            product_category,
            SUM(total_item_rows) AS historical_items
        FROM history
        GROUP BY customer_state, product_category
        ORDER BY historical_items DESC
        LIMIT 20
    )

    SELECT
        h.order_month,
        h.customer_state,
        h.product_category,
        h.total_item_rows AS demand,
        h.delivered_item_rows,
        h.cancelled_item_rows,
        h.unavailable_item_rows,
        h.other_non_cancelled_item_rows,
        t.historical_items
    FROM history h
    INNER JOIN top_combinations t
        ON h.customer_state = t.customer_state
        AND h.product_category = t.product_category

    ORDER BY
        h.customer_state,
        h.product_category,
        h.order_month

) TO '{OUTPUT}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Creating top-20 forecasting dataset...")
con.execute(query)

print("\nDATASET SUMMARY")

print(con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT order_month) AS months,
        COUNT(DISTINCT customer_state || '_' || product_category)
            AS combinations,
        SUM(demand) AS total_demand,
        COUNT(*) FILTER (WHERE demand = 0) AS zero_demand_months,
        MIN(order_month) AS first_month,
        MAX(order_month) AS last_month
    FROM read_parquet('{OUTPUT}')
""").fetchdf().to_string(index=False))

print("\nTOP 20 COMBINATIONS")

print(con.execute(f"""
    SELECT
        customer_state,
        product_category,
        MAX(historical_items) AS historical_items,
        COUNT(*) AS months,
        COUNT(*) FILTER (WHERE demand > 0) AS active_months
    FROM read_parquet('{OUTPUT}')
    GROUP BY 1, 2
    ORDER BY historical_items DESC
""").fetchdf().to_string(index=False))

print("\nSaved:", OUTPUT)
con.close()
