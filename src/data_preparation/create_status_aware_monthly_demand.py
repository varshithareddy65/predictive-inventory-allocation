
import duckdb
from pathlib import Path

INPUT = "datasets/olist_status_aware_demand.parquet"
OUTPUT = "datasets/olist_state_category_status_monthly.parquet"

Path("datasets/duckdb_tmp").mkdir(
    parents=True, exist_ok=True
)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='datasets/duckdb_tmp'")

query = f"""
COPY (
    WITH actual AS (
        SELECT
            order_month,
            customer_state,
            product_category,
            SUM(total_item_rows) AS total_item_rows,
            SUM(delivered_item_rows) AS delivered_item_rows,
            SUM(cancelled_item_rows) AS cancelled_item_rows,
            SUM(unavailable_item_rows) AS unavailable_item_rows,
            SUM(other_non_cancelled_item_rows)
                AS other_non_cancelled_item_rows
        FROM read_parquet('{INPUT}')
        GROUP BY 1, 2, 3
    ),

    months AS (
        SELECT unnest(
            generate_series(
                DATE '2016-09-01',
                DATE '2018-10-01',
                INTERVAL 1 MONTH
            )
        )::DATE AS order_month
    ),

    states AS (
        SELECT DISTINCT customer_state FROM actual
    ),

    categories AS (
        SELECT DISTINCT product_category FROM actual
    ),

    grid AS (
        SELECT
            m.order_month,
            s.customer_state,
            c.product_category
        FROM months m
        CROSS JOIN states s
        CROSS JOIN categories c
    )

    SELECT
        g.order_month,
        g.customer_state,
        g.product_category,

        COALESCE(a.total_item_rows, 0)
            AS total_item_rows,

        COALESCE(a.delivered_item_rows, 0)
            AS delivered_item_rows,

        COALESCE(a.cancelled_item_rows, 0)
            AS cancelled_item_rows,

        COALESCE(a.unavailable_item_rows, 0)
            AS unavailable_item_rows,

        COALESCE(a.other_non_cancelled_item_rows, 0)
            AS other_non_cancelled_item_rows,

        CASE
            WHEN g.order_month IN (
                DATE '2016-09-01',
                DATE '2018-10-01'
            ) THEN 1
            ELSE 0
        END AS is_partial_month

    FROM grid g
    LEFT JOIN actual a
        ON g.order_month = a.order_month
        AND g.customer_state = a.customer_state
        AND g.product_category = a.product_category

) TO '{OUTPUT}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Creating status-aware monthly forecasting dataset...")
con.execute(query)

result = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT order_month) AS months,
        COUNT(DISTINCT customer_state) AS states,
        COUNT(DISTINCT product_category) AS categories,
        SUM(total_item_rows) AS total_items,
        SUM(delivered_item_rows) AS delivered_items,
        SUM(cancelled_item_rows) AS cancelled_items,
        COUNT(*) FILTER (
            WHERE total_item_rows = 0
        ) AS zero_demand_rows,
        COUNT(*) FILTER (
            WHERE is_partial_month = 1
        ) AS partial_month_rows
    FROM read_parquet('{OUTPUT}')
""").fetchone()

names = [
    "rows", "months", "states", "categories",
    "total_item_rows", "delivered_item_rows",
    "cancelled_item_rows", "zero_demand_rows",
    "partial_month_rows"
]

print("\nFORECASTING DATASET CREATED")

for name, value in zip(names, result):
    print(f"{name}: {value}")

print("\nSaved:", OUTPUT)
con.close()
