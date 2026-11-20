
import os
import duckdb

con = duckdb.connect()

# Keep DuckDB within a reasonable memory budget.
con.execute("SET memory_limit = '8GB'")
con.execute("SET threads = 4")

cart_path = "datasets/synerise_dataset/add_to_cart.parquet"
purchase_path = "datasets/synerise_dataset/product_buy.parquet"
properties_path = "datasets/synerise_dataset/product_properties.parquet"
base_path = "datasets/time_features.parquet"

output_path = "datasets/rolling_demand_features.parquet"

os.makedirs("datasets/duckdb_tmp", exist_ok=True)
con.execute("SET temp_directory = 'datasets/duckdb_tmp'")

print("Building Version 2 rolling demand features...")

query = f"""
COPY (
    WITH base AS (
        SELECT
            client_id,
            sku,
            cart_time,
            category
        FROM read_parquet('{base_path}')
    ),

    -- Remove exact duplicate cart events.
    raw_carts AS (
        SELECT DISTINCT
            client_id,
            sku,
            CAST(timestamp AS TIMESTAMP) AS event_time
        FROM read_parquet('{cart_path}')
    ),

    purchases AS (
        SELECT
            client_id,
            sku,
            CAST(timestamp AS TIMESTAMP) AS event_time
        FROM read_parquet('{purchase_path}')
    ),

    -- Product cart counts at each timestamp.
    product_cart_events AS (
        SELECT
            sku,
            event_time,
            COUNT(*) AS cart_count,
            0 AS purchase_count
        FROM raw_carts
        GROUP BY sku, event_time
    ),

    -- Product purchase counts at each timestamp.
    product_purchase_events AS (
        SELECT
            sku,
            event_time,
            0 AS cart_count,
            COUNT(*) AS purchase_count
        FROM purchases
        GROUP BY sku, event_time
    ),

    -- Include both cart and purchase timestamps.
    product_timeline AS (
        SELECT
            sku,
            event_time,
            SUM(cart_count) AS cart_count,
            SUM(purchase_count) AS purchase_count
        FROM (
            SELECT * FROM product_cart_events
            UNION ALL
            SELECT * FROM product_purchase_events
        )
        GROUP BY sku, event_time
    ),

    -- Rolling product demand, strictly before event_time.
    product_rolling AS (
        SELECT
            sku,
            event_time,

            COALESCE(
                SUM(cart_count) OVER (
                    PARTITION BY sku
                    ORDER BY event_time
                    RANGE BETWEEN INTERVAL 7 DAYS PRECEDING
                    AND INTERVAL 1 MICROSECOND PRECEDING
                ), 0
            ) AS product_carts_7d,

            COALESCE(
                SUM(cart_count) OVER (
                    PARTITION BY sku
                    ORDER BY event_time
                    RANGE BETWEEN INTERVAL 30 DAYS PRECEDING
                    AND INTERVAL 1 MICROSECOND PRECEDING
                ), 0
            ) AS product_carts_30d,

            COALESCE(
                SUM(purchase_count) OVER (
                    PARTITION BY sku
                    ORDER BY event_time
                    RANGE BETWEEN INTERVAL 7 DAYS PRECEDING
                    AND INTERVAL 1 MICROSECOND PRECEDING
                ), 0
            ) AS product_purchases_7d,

            COALESCE(
                SUM(purchase_count) OVER (
                    PARTITION BY sku
                    ORDER BY event_time
                    RANGE BETWEEN INTERVAL 30 DAYS PRECEDING
                    AND INTERVAL 1 MICROSECOND PRECEDING
                ), 0
            ) AS product_purchases_30d

        FROM product_timeline
    ),

    -- Map raw cart events to their product categories.
    category_cart_events AS (
        SELECT
            p.category,
            c.event_time,
            COUNT(*) AS cart_count,
            0 AS purchase_count
        FROM raw_carts c
        INNER JOIN read_parquet('{properties_path}') p
            ON c.sku = p.sku
        GROUP BY p.category, c.event_time
    ),

    -- Category purchases at each timestamp.
    category_purchase_events AS (
        SELECT
            p.category,
            b.event_time,
            0 AS cart_count,
            COUNT(*) AS purchase_count
        FROM purchases b
        INNER JOIN read_parquet('{properties_path}') p
            ON b.sku = p.sku
        GROUP BY p.category, b.event_time
    ),

    -- Include cart timestamps as well as purchase timestamps.
    category_timeline AS (
        SELECT
            category,
            event_time,
            SUM(cart_count) AS cart_count,
            SUM(purchase_count) AS purchase_count
        FROM (
            SELECT * FROM category_cart_events
            UNION ALL
            SELECT * FROM category_purchase_events
        )
        GROUP BY category, event_time
    ),

    category_rolling AS (
        SELECT
            category,
            event_time,

            COALESCE(
                SUM(purchase_count) OVER (
                    PARTITION BY category
                    ORDER BY event_time
                    RANGE BETWEEN INTERVAL 7 DAYS PRECEDING
                    AND INTERVAL 1 MICROSECOND PRECEDING
                ), 0
            ) AS category_purchases_7d,

            COALESCE(
                SUM(purchase_count) OVER (
                    PARTITION BY category
                    ORDER BY event_time
                    RANGE BETWEEN INTERVAL 30 DAYS PRECEDING
                    AND INTERVAL 1 MICROSECOND PRECEDING
                ), 0
            ) AS category_purchases_30d

        FROM category_timeline
    ),

    -- Customer purchase history timeline.
    customer_purchase_timeline AS (
        SELECT
            client_id,
            event_time
        FROM purchases
        GROUP BY client_id, event_time
    )

    SELECT
        b.client_id,
        b.sku,
        b.cart_time,

        COALESCE(pr.product_carts_7d, 0)
            AS product_carts_7d,

        COALESCE(pr.product_carts_30d, 0)
            AS product_carts_30d,

        COALESCE(pr.product_purchases_7d, 0)
            AS product_purchases_7d,

        COALESCE(pr.product_purchases_30d, 0)
            AS product_purchases_30d,

        COALESCE(cr.category_purchases_7d, 0)
            AS category_purchases_7d,

        COALESCE(cr.category_purchases_30d, 0)
            AS category_purchases_30d,

        CASE
            WHEN cp.event_time IS NOT NULL
            THEN DATE_DIFF(
                'microsecond',
                cp.event_time,
                b.cart_time
            ) / 86400000000.0
            ELSE NULL
        END AS customer_purchase_recency_days,

        CASE
            WHEN pp.event_time IS NOT NULL
            THEN DATE_DIFF(
                'microsecond',
                pp.event_time,
                b.cart_time
            ) / 86400000000.0
            ELSE NULL
        END AS product_purchase_recency_days

    FROM base b

    LEFT JOIN product_rolling pr
        ON b.sku = pr.sku
        AND b.cart_time = pr.event_time

    LEFT JOIN category_rolling cr
        ON b.category = cr.category
        AND b.cart_time = cr.event_time

    ASOF LEFT JOIN customer_purchase_timeline cp
        ON b.client_id = cp.client_id
        AND b.cart_time > cp.event_time

    ASOF LEFT JOIN purchases pp
        ON b.sku = pp.sku
        AND b.cart_time > pp.event_time

) TO '{output_path}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

con.execute(query)

print("\nFEATURE GENERATION COMPLETED")

summary = con.execute(f"""
SELECT
    COUNT(*) AS total_rows,

    AVG(product_carts_7d) AS avg_product_carts_7d,
    AVG(product_carts_30d) AS avg_product_carts_30d,

    AVG(product_purchases_7d)
        AS avg_product_purchases_7d,

    AVG(product_purchases_30d)
        AS avg_product_purchases_30d,

    AVG(category_purchases_7d)
        AS avg_category_purchases_7d,

    AVG(category_purchases_30d)
        AS avg_category_purchases_30d,

    AVG(customer_purchase_recency_days)
        AS avg_customer_recency_days,

    AVG(product_purchase_recency_days)
        AS avg_product_recency_days,

    COUNT(*) FILTER (
        WHERE customer_purchase_recency_days IS NULL
    ) AS customers_without_previous_purchase

FROM read_parquet('{output_path}')
""").fetchdf()

print(summary.to_string(index=False))

print("\nSAMPLE FEATURES")

print(con.execute(f"""
SELECT *
FROM read_parquet('{output_path}')
LIMIT 10
""").fetchdf().to_string(index=False))

print("\nSaved:", output_path)

con.close()
