
import duckdb
from pathlib import Path

DATA = "datasets/olist/"
OUTPUT = "datasets/olist_geographic_demand.parquet"

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

    delivered_orders AS (
        SELECT
            order_id,
            customer_id,
            CAST(order_purchase_timestamp AS TIMESTAMP)
                AS purchase_timestamp
        FROM read_csv_auto(
            '{DATA}olist_orders_dataset.csv'
        )
        WHERE order_status = 'delivered'
          AND order_purchase_timestamp IS NOT NULL
    ),

    order_details AS (
        SELECT
            o.order_id,
            o.purchase_timestamp,
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
        FROM delivered_orders o
        JOIN read_csv_auto(
            '{DATA}olist_order_items_dataset.csv'
        ) i ON o.order_id = i.order_id
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
        DATE_TRUNC('month', d.purchase_timestamp)
            AS order_month,
        d.zip_prefix,
        d.customer_city,
        d.customer_state,
        d.product_category,

        g.latitude,
        g.longitude,

        COUNT(DISTINCT d.order_id)
            AS order_count,

        COUNT(DISTINCT d.customer_unique_id)
            AS unique_customer_count,

        COUNT(*) AS order_item_rows,

        COUNT(DISTINCT d.product_id)
            AS distinct_products,

        SUM(d.price) AS product_sales_value

    FROM order_details d
    LEFT JOIN geo g
        ON d.zip_prefix = g.zip_prefix

    GROUP BY
        1, 2, 3, 4, 5, 6, 7

) TO '{OUTPUT}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Building geographic demand dataset...")
con.execute(query)

result = con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(*) FILTER (
            WHERE latitude IS NULL OR longitude IS NULL
        ) AS rows_without_coordinates,
        COUNT(DISTINCT zip_prefix) AS zip_prefixes,
        COUNT(DISTINCT order_month) AS months,
        SUM(order_item_rows) AS total_item_rows,
        SUM(product_sales_value) AS total_product_sales
    FROM read_parquet('{OUTPUT}')
""").fetchone()

print("\nGEOGRAPHIC DEMAND DATASET CREATED")

names = [
    "output_rows",
    "rows_without_coordinates",
    "distinct_zip_prefixes",
    "months",
    "total_order_item_rows",
    "total_product_sales_value"
]

for name, value in zip(names, result):
    print(f"{name}: {value}")

print("\nSaved:", OUTPUT)

con.close()
