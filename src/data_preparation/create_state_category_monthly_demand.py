
import duckdb
from pathlib import Path

INPUT = "datasets/olist_geographic_demand.parquet"
OUTPUT = "datasets/olist_state_category_monthly.parquet"

Path("datasets/duckdb_tmp").mkdir(
    parents=True, exist_ok=True
)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='datasets/duckdb_tmp'")

query = f"""
COPY (
    WITH monthly_actual AS (
        SELECT
            CAST(order_month AS DATE) AS order_month,
            customer_state,
            product_category,
            SUM(order_item_rows) AS order_item_rows,
            SUM(product_sales_value) AS product_sales_value
        FROM read_parquet('{INPUT}')
        GROUP BY 1, 2, 3
    ),

    states AS (
        SELECT DISTINCT customer_state
        FROM monthly_actual
    ),

    categories AS (
        SELECT DISTINCT product_category
        FROM monthly_actual
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

    complete_grid AS (
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

        COALESCE(a.order_item_rows, 0)
            AS order_item_rows,

        COALESCE(a.product_sales_value, 0)
            AS product_sales_value,

        CASE
            WHEN g.order_month IN (
                DATE '2016-09-01',
                DATE '2018-10-01'
            ) THEN 1
            ELSE 0
        END AS is_partial_month

    FROM complete_grid g
    LEFT JOIN monthly_actual a
        ON g.order_month = a.order_month
        AND g.customer_state = a.customer_state
        AND g.product_category = a.product_category

) TO '{OUTPUT}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Creating monthly state-category demand dataset...")
con.execute(query)

summary = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT customer_state) AS states,
        COUNT(DISTINCT product_category) AS categories,
        COUNT(DISTINCT order_month) AS months,
        SUM(order_item_rows) AS total_item_rows,
        SUM(product_sales_value) AS total_sales,
        COUNT(*) FILTER (
            WHERE order_item_rows = 0
        ) AS zero_demand_rows,
        COUNT(*) FILTER (
            WHERE is_partial_month = 1
        ) AS partial_month_rows
    FROM read_parquet('{OUTPUT}')
""").fetchone()

names = [
    "total_rows",
    "states",
    "categories",
    "months",
    "total_item_rows",
    "total_product_sales_value",
    "zero_demand_rows",
    "partial_month_rows"
]

print("\nMONTHLY DEMAND DATASET CREATED")

for name, value in zip(names, summary):
    print(f"{name}: {value}")

print("\nSaved:", OUTPUT)

con.close()
