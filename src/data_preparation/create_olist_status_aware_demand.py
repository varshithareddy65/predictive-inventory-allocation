
import duckdb
from pathlib import Path

DATA = "datasets/olist/"
OUTPUT = "datasets/olist_status_aware_demand.parquet"

Path("datasets/duckdb_tmp").mkdir(
    parents=True, exist_ok=True
)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='datasets/duckdb_tmp'")

query = f"""
COPY (
    WITH geo AS (
        SELECT
            LPAD(
                CAST(geolocation_zip_code_prefix AS VARCHAR),
                5, '0'
            ) AS zip_prefix,
            AVG(geolocation_lat) AS latitude,
            AVG(geolocation_lng) AS longitude
        FROM read_csv_auto(
            '{DATA}olist_geolocation_dataset.csv'
        )
        WHERE geolocation_lat IS NOT NULL
          AND geolocation_lng IS NOT NULL
        GROUP BY 1
    ),

    order_details AS (
        SELECT
            o.order_id,
            DATE_TRUNC(
                'month',
                CAST(o.order_purchase_timestamp AS TIMESTAMP)
            )::DATE AS order_month,
            o.order_status,

            c.customer_unique_id,

            LPAD(
                CAST(c.customer_zip_code_prefix AS VARCHAR),
                5, '0'
            ) AS zip_prefix,

            c.customer_city,
            c.customer_state,

            COALESCE(
                t.product_category_name_english,
                p.product_category_name,
                'unknown'
            ) AS product_category,

            i.product_id,
            i.price

        FROM read_csv_auto(
            '{DATA}olist_orders_dataset.csv'
        ) o

        JOIN read_csv_auto(
            '{DATA}olist_order_items_dataset.csv'
        ) i USING (order_id)

        JOIN read_csv_auto(
            '{DATA}olist_customers_dataset.csv'
        ) c ON o.customer_id = c.customer_id

        LEFT JOIN read_csv_auto(
            '{DATA}olist_products_dataset.csv'
        ) p ON i.product_id = p.product_id

        LEFT JOIN read_csv_auto(
            '{DATA}product_category_name_translation.csv'
        ) t ON p.product_category_name =
              t.product_category_name
    )

    SELECT
        d.order_month,
        d.customer_state,
        d.customer_city,
        d.zip_prefix,
        d.product_category,

        g.latitude,
        g.longitude,

        COUNT(DISTINCT d.order_id) AS total_orders,

        COUNT(DISTINCT d.customer_unique_id)
            AS unique_customers,

        COUNT(*) AS total_item_rows,

        COUNT(*) FILTER (
            WHERE d.order_status = 'delivered'
        ) AS delivered_item_rows,

        COUNT(*) FILTER (
            WHERE d.order_status = 'canceled'
        ) AS cancelled_item_rows,

        COUNT(*) FILTER (
            WHERE d.order_status = 'unavailable'
        ) AS unavailable_item_rows,

        COUNT(*) FILTER (
            WHERE d.order_status NOT IN (
                'delivered',
                'canceled',
                'unavailable'
            )
        ) AS other_non_cancelled_item_rows,

        COUNT(*) FILTER (
            WHERE d.order_status != 'canceled'
        ) AS recorded_non_cancelled_items,

        SUM(d.price) AS total_product_sales_value,

        SUM(d.price) FILTER (
            WHERE d.order_status = 'delivered'
        ) AS delivered_product_sales_value

    FROM order_details d

    LEFT JOIN geo g
        ON d.zip_prefix = g.zip_prefix

    GROUP BY
        1, 2, 3, 4, 5, 6, 7

) TO '{OUTPUT}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Building status-aware geographic demand dataset...")

con.execute(query)

result = con.execute(f"""
    SELECT
        COUNT(*) AS output_rows,
        COUNT(DISTINCT order_month) AS months,
        COUNT(DISTINCT customer_state) AS states,
        COUNT(DISTINCT product_category) AS categories,

        SUM(total_item_rows) AS total_items,
        SUM(delivered_item_rows) AS delivered_items,
        SUM(cancelled_item_rows) AS cancelled_items,
        SUM(unavailable_item_rows) AS unavailable_items,
        SUM(other_non_cancelled_item_rows)
            AS other_non_cancelled_items,

        SUM(recorded_non_cancelled_items)
            AS non_cancelled_items,

        COUNT(*) FILTER (
            WHERE latitude IS NULL
        ) AS rows_without_coordinates

    FROM read_parquet('{OUTPUT}')
""").fetchone()

names = [
    "output_rows",
    "months",
    "states",
    "categories",
    "total_item_rows",
    "delivered_item_rows",
    "cancelled_item_rows",
    "unavailable_item_rows",
    "other_non_cancelled_item_rows",
    "recorded_non_cancelled_items",
    "rows_without_coordinates"
]

print("\nSTATUS-AWARE DATASET CREATED")

for name, value in zip(names, result):
    print(f"{name}: {value}")

print("\nSaved:", OUTPUT)

con.close()
