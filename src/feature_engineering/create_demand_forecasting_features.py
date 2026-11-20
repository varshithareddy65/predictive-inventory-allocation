
import duckdb
from pathlib import Path

INPUT = "datasets/forecasting_experiment_top20.parquet"
OUTPUT = "datasets/demand_forecasting_features.parquet"

Path("datasets/duckdb_tmp").mkdir(
    parents=True, exist_ok=True
)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='datasets/duckdb_tmp'")

query = f"""
COPY (
    WITH base AS (
        SELECT
            order_month,
            customer_state,
            product_category,
            demand,
            EXTRACT(MONTH FROM order_month)::INTEGER AS month
        FROM read_parquet('{INPUT}')
    ),

    features AS (
        SELECT
            *,
            LAG(demand, 1) OVER w AS lag_1,
            LAG(demand, 2) OVER w AS lag_2,
            LAG(demand, 3) OVER w AS lag_3,
            LAG(demand, 6) OVER w AS lag_6,

            AVG(demand) OVER (
                PARTITION BY customer_state, product_category
                ORDER BY order_month
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) AS rolling_mean_3,

            AVG(demand) OVER (
                PARTITION BY customer_state, product_category
                ORDER BY order_month
                ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
            ) AS rolling_mean_6,

            STDDEV_SAMP(demand) OVER (
                PARTITION BY customer_state, product_category
                ORDER BY order_month
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) AS rolling_std_3

        FROM base
        WINDOW w AS (
            PARTITION BY customer_state, product_category
            ORDER BY order_month
        )
    )

    SELECT *
    FROM features
    WHERE lag_6 IS NOT NULL

    ORDER BY customer_state, product_category, order_month

) TO '{OUTPUT}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Creating forecasting features...")
con.execute(query)

print("\nFEATURE DATASET SUMMARY")

print(con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT customer_state || '_' || product_category)
            AS combinations,
        COUNT(DISTINCT order_month) AS months,
        MIN(order_month) AS first_month,
        MAX(order_month) AS last_month,

        COUNT(*) FILTER (WHERE lag_1 IS NULL)
            AS missing_lag_1,

        COUNT(*) FILTER (WHERE rolling_mean_3 IS NULL)
            AS missing_rolling_mean_3,

        COUNT(*) FILTER (WHERE rolling_std_3 IS NULL)
            AS missing_rolling_std_3

    FROM read_parquet('{OUTPUT}')
""").fetchdf().to_string(index=False))

print("\nSaved:", OUTPUT)
con.close()
