from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"

st.set_page_config(
    page_title="Smart Inventory Allocation",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. DATA LOADING
# =========================================================
@st.cache_data
def load_parquet(filename):
    path = DATA_DIR / filename
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    return pd.read_csv(path) if path.exists() else None


historical = load_parquet("olist_state_category_monthly.parquet")
forecasts = load_parquet("hybrid_forecasting_predictions.parquet")
allocation_v1 = load_parquet("inventory_allocation_recommendations.parquet")
allocation_v2 = load_parquet("inventory_allocation_recommendations_v2.parquet")
forecast_metrics = load_csv("hybrid_performance_comparison.csv")
baseline_metrics = load_parquet("baseline_forecasting_results.parquet")
geographic = load_parquet("olist_geographic_demand.parquet")


def show_missing(message):
    st.warning(message)


def format_number(value):
    return f"{value:,.2f}"


def safe_sum(df, column):
    return float(df[column].sum()) if df is not None and column in df else 0.0


# =========================================================
# 3. SIDEBAR
# =========================================================
st.sidebar.title("📦 Smart Inventory System")
st.sidebar.caption("Predictive Demand & Inventory Allocation")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Demand Analysis",
        "Geographical Demand",
        "Demand Forecasting",
        "Inventory Allocation",
        "Model Performance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Data: Olist Brazilian E-Commerce Dataset")
st.sidebar.caption("Inventory availability: Simulated assumption")


# =========================================================
# 4. EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":

    st.title("Smart Inventory Allocation Dashboard")

    st.markdown(
        "Predictive demand forecasting and inventory allocation "
        "across customer delivery regions and product categories."
    )

    st.info(
        "Historical demand is derived from the Olist Brazilian E-Commerce "
        "Dataset. Inventory availability is simulated at 80% of estimated "
        "category requirements and does not represent actual warehouse stock."
    )

    if forecasts is None or allocation_v2 is None:

        show_missing(
            "Forecast or allocation data is missing. Check the datasets folder."
        )

    else:

        forecast_df = forecasts.copy()
        allocation_df = allocation_v2.copy()

        forecast_df["forecast_month"] = pd.to_datetime(
            forecast_df["forecast_month"]
        )

        # Latest forecast month
        latest_forecast_month = forecast_df["forecast_month"].max()

        latest_forecast_df = forecast_df[
            forecast_df["forecast_month"] == latest_forecast_month
        ].copy()

        # KPI calculations
        total_forecast = safe_sum(latest_forecast_df, "prediction")
        total_required = safe_sum(allocation_df, "required_stock")
        total_allocated = safe_sum(allocation_df, "allocated_stock")
        total_unmet = safe_sum(allocation_df, "unmet_forecast_demand")
        total_shortage = safe_sum(allocation_df, "shortage")
        total_remaining_safety = safe_sum(
            allocation_df, "remaining_safety_stock"
        )

        requirement_coverage = (
            total_allocated / total_required * 100
            if total_required > 0 else 0
        )

        forecast_coverage = (
            max(total_forecast - total_unmet, 0) / total_forecast * 100
            if total_forecast > 0 else 0
        )

        st.caption(
            f"Allocation KPIs and forecast charts refer to "
            f"{latest_forecast_month:%B %Y}."
        )

        st.subheader("Inventory and Demand Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Forecast Demand", f"{total_forecast:,.0f}")
        c2.metric("Required Stock", f"{total_required:,.0f}")
        c3.metric("Allocated Stock", f"{total_allocated:,.0f}")
        c4.metric("Unmet Forecast Demand", f"{total_unmet:,.2f}")

        c5, c6, c7 = st.columns(3)

        c5.metric("Requirement Coverage", f"{requirement_coverage:.2f}%")
        c6.metric("Forecast Demand Coverage", f"{forecast_coverage:.2f}%")
        c7.metric("Remaining Safety Stock", f"{total_remaining_safety:,.2f}")

        st.divider()

        # Category Forecast
        st.subheader("Forecast Demand by Product Category")

        category_forecast = (
            latest_forecast_df.groupby(
                "product_category", as_index=False
            )["prediction"]
            .sum()
            .sort_values("prediction", ascending=False)
        )

        fig = px.bar(
            category_forecast,
            x="product_category",
            y="prediction",
            title=f"{latest_forecast_month:%B %Y} Forecast Demand by Product Category",
            labels={
                "product_category": "Product Category",
                "prediction": "Forecast Demand",
            },
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=450,
        )

        st.plotly_chart(fig, width="stretch")

        st.divider()

        # State Forecast
        st.subheader("Forecast Demand by Customer State")

        state_forecast = (
            latest_forecast_df.groupby(
                "customer_state", as_index=False
            )["prediction"]
            .sum()
            .sort_values("prediction", ascending=False)
        )

        fig = px.bar(
            state_forecast,
            x="customer_state",
            y="prediction",
            title=f"{latest_forecast_month:%B %Y} Forecast Demand by Customer Delivery State",
            labels={
                "customer_state": "Customer Delivery State",
                "prediction": "Forecast Demand",
            },
        )

        st.plotly_chart(fig, width="stretch")

        st.divider()

        # Allocation Status
        st.subheader("Inventory Allocation Status")

        left, right = st.columns(2)

        with left:

            allocation_status = pd.DataFrame({
                "Status": [
                    "Allocated Stock",
                    "Requirement Shortage",
                ],
                "Stock": [
                    total_allocated,
                    total_shortage,
                ],
            })

            fig = px.pie(
                allocation_status,
                names="Status",
                values="Stock",
                hole=0.5,
                title="Allocated Stock vs Requirement Shortage",
            )

            st.plotly_chart(fig, width="stretch")

        with right:

            demand_status = pd.DataFrame({
                "Status": [
                    "Covered Forecast Demand",
                    "Unmet Forecast Demand",
                ],
                "Demand": [
                    max(total_forecast - total_unmet, 0),
                    total_unmet,
                ],
            })

            fig = px.pie(
                demand_status,
                names="Status",
                values="Demand",
                hole=0.5,
                title="Forecast Demand Coverage",
            )

            st.plotly_chart(fig, width="stretch")

        st.divider()

        # V1 vs V2
        st.subheader("Allocation Strategy Comparison")

        if allocation_v1 is not None:

            v1 = allocation_v1.copy()
            v2 = allocation_df.copy()

            v1_unmet = safe_sum(v1, "unmet_forecast_demand")
            v2_unmet = safe_sum(v2, "unmet_forecast_demand")

            comparison = pd.DataFrame({
                "Method": ["V1", "V2"],
                "Allocated Stock": [
                    safe_sum(v1, "allocated_stock"),
                    safe_sum(v2, "allocated_stock"),
                ],
                "Unmet Forecast Demand": [
                    v1_unmet,
                    v2_unmet,
                ],
            })

            left, right = st.columns(2)

            with left:

                fig = px.bar(
                    comparison,
                    x="Method",
                    y="Allocated Stock",
                    title="Allocated Stock Comparison",
                    text_auto=".0f",
                )

                st.plotly_chart(fig, width="stretch")

            with right:

                fig = px.bar(
                    comparison,
                    x="Method",
                    y="Unmet Forecast Demand",
                    title="Unmet Forecast Demand Comparison",
                    text_auto=".2f",
                )

                st.plotly_chart(fig, width="stretch")

            improvement = (
                (v1_unmet - v2_unmet) / v1_unmet * 100
                if v1_unmet > 0 else 0
            )

            st.success(
                f"V2 reduces unmet forecast demand by "
                f"{v1_unmet - v2_unmet:,.2f} "
                f"compared with V1 ({improvement:.2f}% reduction)."
            )

        else:

            st.warning("V1 allocation results were not found.")

        st.divider()

        # Summary
        st.subheader("Allocation Summary")

        summary = pd.DataFrame({
            "Metric": [
                "Forecast Demand",
                "Required Stock",
                "Allocated Stock",
                "Requirement Shortage",
                "Unmet Forecast Demand",
                "Remaining Safety Stock",
            ],
            "Value": [
                round(total_forecast, 2),
                round(total_required, 2),
                round(total_allocated, 2),
                round(total_shortage, 2),
                round(total_unmet, 2),
                round(total_remaining_safety, 2),
            ],
        })

        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
        )

        st.divider()

        st.subheader("Project Methodology")

        st.markdown("""
- **Historical demand:** Olist delivered order-item records.
- **Forecasting:** Naive and Holt methods evaluated using chronological validation and test periods.
- **Model selection:** Hybrid method selects a model separately for each state-category combination using validation-period WAPE.
- **Safety stock:** Estimated from validation forecast errors using a heuristic 90% service-level factor.
- **V1 allocation:** Proportional allocation of simulated available stock across requirements.
- **V2 allocation:** Prioritizes forecast demand before distributing remaining stock toward safety stock.
        """)

        st.caption(
            "Forecast evaluation covers January–August 2018. These are historical "
            "test-period predictions, not live future forecasts. Customer states "
            "represent delivery-demand regions, not verified warehouse locations."
        )
# =========================================================
# 5. DEMAND ANALYSIS
# =========================================================
elif page == "Demand Analysis":

    st.title("Historical Demand Analysis")

    if historical is None:

        show_missing("olist_state_category_monthly.parquet not found.")

    else:

        df = historical.copy()

        df["order_month"] = pd.to_datetime(df["order_month"])

        st.markdown(
            "Explore historical delivered order-item records by month, "
            "customer state and product category."
        )

        states = sorted(
            df["customer_state"].dropna().unique()
        )

        selected_states = st.multiselect(
            "Select customer states",
            states,
            default=states,
        )

        categories = sorted(
            df["product_category"].dropna().unique()
        )

        top_categories = (
            df.groupby("product_category")["order_item_rows"]
            .sum()
            .nlargest(5)
            .index.tolist()
        )

        selected_categories = st.multiselect(
            "Select product categories",
            categories,
            default=top_categories,
        )

        filtered = df[
            df["customer_state"].isin(selected_states)
            & df["product_category"].isin(selected_categories)
        ]

        total_items = safe_sum(
            filtered,
            "order_item_rows",
        )

        st.metric(
            "Historical Order-Item Records",
            f"{total_items:,.0f}",
        )

        # Monthly Demand
        monthly = (
            filtered.groupby(
                "order_month",
                as_index=False,
            )["order_item_rows"]
            .sum()
        )

        fig = px.line(
            monthly,
            x="order_month",
            y="order_item_rows",
            markers=True,
            title="Monthly Historical Demand",
            labels={
                "order_month": "Month",
                "order_item_rows": "Order-Item Records",
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        # State Demand
        state_demand = (
            filtered.groupby(
                "customer_state",
                as_index=False,
            )["order_item_rows"]
            .sum()
            .sort_values(
                "order_item_rows",
                ascending=False,
            )
        )

        fig = px.bar(
            state_demand,
            x="customer_state",
            y="order_item_rows",
            title="Demand by Customer State",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.subheader("Demand Data")

        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
        )


# =========================================================
# 6. GEOGRAPHICAL DEMAND
# =========================================================
elif page == "Geographical Demand":

    st.title("Geographical Demand Distribution")

    st.markdown(
        "Explore historical customer demand across Brazilian delivery "
        "regions using ZIP-code-level geographic data."
    )

    if geographic is None:

        show_missing(
            "olist_geographic_demand.parquet not found."
        )

    else:

        geo = geographic.copy()

        geo["order_month"] = pd.to_datetime(
            geo["order_month"]
        )

        states = sorted(
            geo["customer_state"].dropna().unique()
        )

        categories = sorted(
            geo["product_category"].dropna().unique()
        )

        top_states = (
            geo.groupby("customer_state")["order_item_rows"]
            .sum()
            .nlargest(5)
            .index.tolist()
        )

        selected_states = st.multiselect(
            "Select customer states",
            states,
            default=top_states,
        )

        top_categories = (
            geo.groupby("product_category")["order_item_rows"]
            .sum()
            .nlargest(5)
            .index.tolist()
        )

        selected_categories = st.multiselect(
            "Select product categories",
            categories,
            default=top_categories,
        )

        filtered = geo[
            geo["customer_state"].isin(selected_states)
            & geo["product_category"].isin(selected_categories)
        ].copy()

        total_demand = safe_sum(
            filtered,
            "order_item_rows",
        )

        total_orders = safe_sum(
            filtered,
            "order_count",
        )

        total_customers = safe_sum(
            filtered,
            "unique_customer_count",
        )

        mapped = filtered.dropna(
            subset=["latitude", "longitude"]
        ).copy()

        missing_coordinates = (
            len(filtered) - len(mapped)
        )

        # KPIs
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Historical Item Records",
            f"{total_demand:,.0f}",
        )

        col2.metric(
            "Order Records",
            f"{total_orders:,.0f}",
        )

        col3.metric(
            "ZIP Regions on Map",
            f"{mapped['zip_prefix'].nunique():,}",
        )

        col4.metric(
            "Rows Without Coordinates",
            f"{missing_coordinates:,}",
        )

        st.divider()

        # Aggregate by ZIP prefix
        zip_demand = (
            mapped.groupby(
                [
                    "zip_prefix",
                    "customer_city",
                    "customer_state",
                ],
                as_index=False,
            )
            .agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                order_item_rows=("order_item_rows", "sum"),
                order_count=("order_count", "sum"),
                unique_customer_count=(
                    "unique_customer_count",
                    "sum",
                ),
                product_sales_value=(
                    "product_sales_value",
                    "sum",
                ),
            )
        )

        st.subheader("Customer Demand Map")

        st.caption(
            "Each circle represents a ZIP-code region. "
            "Larger circles indicate more historical order-item records."
        )

        if zip_demand.empty:

            st.warning(
                "No geographic records are available for the selected filters."
            )

        else:

            fig = px.scatter_map(
                zip_demand,
                lat="latitude",
                lon="longitude",
                size="order_item_rows",
                color="order_item_rows",
                hover_name="customer_city",
                hover_data={
                    "zip_prefix": True,
                    "customer_state": True,
                    "order_item_rows": True,
                    "order_count": True,
                    "unique_customer_count": True,
                    "product_sales_value": ":,.2f",
                    "latitude": False,
                    "longitude": False,
                },
                size_max=18,
                zoom=3.2,
                center={
                    "lat": -14.2,
                    "lon": -51.9,
                },
                map_style="open-street-map",
                color_continuous_scale="Blues",
                title="Historical Demand Across Brazil",
            )

            fig.update_layout(
                height=650,
                margin={
                    "l": 0,
                    "r": 0,
                    "t": 45,
                    "b": 0,
                },
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        # State Demand
        st.subheader("Demand by Customer State")

        state_demand = (
            filtered.groupby(
                "customer_state",
                as_index=False,
            )["order_item_rows"]
            .sum()
            .sort_values(
                "order_item_rows",
                ascending=False,
            )
        )

        fig = px.bar(
            state_demand,
            x="customer_state",
            y="order_item_rows",
            color="order_item_rows",
            color_continuous_scale="Blues",
            title="Historical Order-Item Records by State",
            labels={
                "customer_state": "Customer State",
                "order_item_rows": "Order-Item Records",
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        # ZIP Details
        st.subheader("ZIP-Code-Level Demand Details")

        st.dataframe(
            zip_demand.sort_values(
                "order_item_rows",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Geographic locations represent customer delivery regions, "
            "not verified warehouse or hub locations. "
            "The map aggregates historical records by ZIP prefix."
        )


# =========================================================
# 7. DEMAND FORECASTING
# =========================================================
elif page == "Demand Forecasting":

    st.title("Demand Forecasting")

    if forecasts is None:

        show_missing(
            "hybrid_forecasting_predictions.parquet not found."
        )

    else:

        df = forecasts.copy()

        df["forecast_month"] = pd.to_datetime(
            df["forecast_month"]
        )

        st.markdown(
            "Compare actual demand with model predictions "
            "for the selected state-category combination."
        )

        combinations = (
            df[
                [
                    "customer_state",
                    "product_category",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "customer_state",
                    "product_category",
                ]
            )
        )

        options = [
            f"{row.customer_state} - {row.product_category}"
            for row in combinations.itertuples(index=False)
        ]

        selected = st.selectbox(
            "Select state and category",
            options,
            index=0,
        )

        selected_state, selected_category = selected.split(
            " - ",
            1,
        )

        subset = df[
            (df["customer_state"] == selected_state)
            & (df["product_category"] == selected_category)
        ].sort_values(
            "forecast_month"
        )

        fig = px.line(
            subset,
            x="forecast_month",
            y=[
                "actual",
                "prediction",
            ],
            markers=True,
            title=(
                f"Actual vs Predicted Demand: "
                f"{selected_state} - {selected_category}"
            ),
            labels={
                "forecast_month": "Month",
                "value": "Order-Item Records",
                "variable": "Series",
            },
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.subheader("Forecast Results")

        st.dataframe(
            subset,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "These are rolling one-step-ahead test forecasts "
            "from January to August 2018, not live future predictions."
        )    
# =========================================================
# 8. INVENTORY ALLOCATION
# =========================================================
elif page == "Inventory Allocation":

    st.title("Inventory Allocation Analysis")

    if allocation_v1 is None or allocation_v2 is None:

        show_missing(
            "Both V1 and V2 allocation result files are required."
        )

    else:

        v1 = allocation_v1.copy()
        v2 = allocation_v2.copy()

        v1_total = safe_sum(
            v1,
            "unmet_forecast_demand",
        )

        v2_total = safe_sum(
            v2,
            "unmet_forecast_demand",
        )

        improvement = (
            (v1_total - v2_total) / v1_total * 100
            if v1_total > 0 else 0
        )

        # -------------------------------------------------
        # PERFORMANCE KPIs
        # -------------------------------------------------
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "V1 Unmet Demand",
            format_number(v1_total),
        )

        c2.metric(
            "V2 Unmet Demand",
            format_number(v2_total),
        )

        c3.metric(
            "Reduction in Unmet Demand",
            f"{improvement:.2f}%",
        )

        st.divider()

        # -------------------------------------------------
        # V1 VS V2 COMPARISON
        # -------------------------------------------------
        st.subheader("V1 vs V2 Allocation")

        comparison = pd.DataFrame({
            "Method": ["V1", "V2"],

            "Allocated Stock": [
                safe_sum(v1, "allocated_stock"),
                safe_sum(v2, "allocated_stock"),
            ],

            "Requirement Shortage": [
                safe_sum(v1, "shortage"),
                safe_sum(v2, "shortage"),
            ],

            "Unmet Forecast Demand": [
                v1_total,
                v2_total,
            ],

            "Remaining Safety Stock": [
                safe_sum(v1, "remaining_safety_stock"),
                safe_sum(v2, "remaining_safety_stock"),
            ],
        })

        fig = px.bar(
            comparison,
            x="Method",
            y=[
                "Allocated Stock",
                "Requirement Shortage",
            ],
            barmode="group",
            title="Stock Allocation Comparison",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        # -------------------------------------------------
        # CATEGORY-WISE UNMET DEMAND
        # -------------------------------------------------
        st.subheader("Unmet Forecast Demand by Category")

        category_comparison = (
            v1.groupby("product_category")[
                "unmet_forecast_demand"
            ]
            .sum()
            .rename("V1")
            .to_frame()
            .join(
                v2.groupby("product_category")[
                    "unmet_forecast_demand"
                ]
                .sum()
                .rename("V2")
            )
            .reset_index()
        )

        category_long = category_comparison.melt(
            id_vars="product_category",
            var_name="Method",
            value_name="Unmet Forecast Demand",
        )

        fig = px.bar(
            category_long,
            x="product_category",
            y="Unmet Forecast Demand",
            color="Method",
            barmode="group",
            title="Unmet Demand by Category",
        )

        fig.update_layout(
            xaxis_tickangle=-45,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.divider()

        # -------------------------------------------------
        # STATE-CATEGORY ALLOCATION DETAILS
        # -------------------------------------------------
        st.subheader("State-Category Allocation Details")

        method = st.radio(
            "Choose allocation method",
            ["V1", "V2"],
            horizontal=True,
        )

        selected_data = (
            v1 if method == "V1" else v2
        )

        st.dataframe(
            selected_data,
            width="stretch",
            hide_index=True,
        )

        # -------------------------------------------------
        # DOWNLOAD CSV
        # -------------------------------------------------
        st.download_button(
            "Download Allocation CSV",
            data=selected_data.to_csv(index=False),
            file_name=f"inventory_allocation_{method.lower()}.csv",
            mime="text/csv",
        )

        st.info(
            "V2 prioritizes forecast demand before distributing "
            "remaining stock toward safety stock. Both methods "
            "use simulated category-level stock availability."
        )        
# =========================================================
# 9. MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.title("Forecasting Model Performance")

    st.markdown(
        "Evaluate forecasting methods using chronological "
        "validation and final test experiments."
    )

    if baseline_metrics is None or forecast_metrics is None:

        show_missing(
            "Baseline or hybrid forecasting results are missing "
            "from the datasets folder."
        )

    else:

        baseline_df = baseline_metrics.copy()
        comparison_df = forecast_metrics.copy()

        # -------------------------------------------------
        # BASELINE FORECASTING RESULTS
        # -------------------------------------------------
        st.header("Baseline Forecasting Results")

        st.markdown(
            "Three baseline methods were evaluated using "
            "rolling one-step-ahead forecasting."
        )

        baseline_summary = (
            baseline_df.groupby("method")
            .apply(
                lambda x: pd.Series({
                    "MAE": (
                        x["actual"] - x["prediction"]
                    ).abs().mean(),

                    "RMSE": (
                        (
                            x["actual"] - x["prediction"]
                        ) ** 2
                    ).mean() ** 0.5,

                    "WAPE (%)": (
                        (
                            x["actual"] - x["prediction"]
                        ).abs().sum()
                        / x["actual"].abs().sum() * 100
                    ),
                }),
                include_groups=False,
            )
            .reset_index()
            .sort_values("WAPE (%)")
        )

        st.dataframe(
            baseline_summary.round(3),
            width="stretch",
            hide_index=True,
        )

        fig = px.bar(
            baseline_summary,
            x="method",
            y="WAPE (%)",
            color="WAPE (%)",
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="Baseline Models — WAPE Comparison",
        )

        fig.update_layout(
            height=420,
            xaxis_title="Forecasting Method",
            yaxis_title="WAPE (%)",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.info(
            "The Naive baseline achieved the best overall "
            "baseline performance, with a WAPE of approximately 18.92%."
        )

        st.divider()

        # -------------------------------------------------
        # HYBRID FORECASTING PERFORMANCE
        # -------------------------------------------------
        st.header("Hybrid Forecasting Performance")

        st.markdown(
            "The hybrid approach selects either Naive or Holt "
            "separately for each state-category combination. "
            "Selection is based on validation-period WAPE."
        )

        required_columns = [
            "customer_state",
            "product_category",
            "hybrid_wape",
            "naive_wape",
            "holt_wape",
            "best_model",
            "hybrid_vs_naive",
        ]

        display_comparison = comparison_df[
            required_columns
        ].copy()

        display_comparison.columns = [
            "Customer State",
            "Product Category",
            "Hybrid WAPE (%)",
            "Naive WAPE (%)",
            "Holt WAPE (%)",
            "Selected Model",
            "Hybrid vs Naive (pp)",
        ]

        st.dataframe(
            display_comparison.round(2),
            width="stretch",
            hide_index=True,
        )

        # -------------------------------------------------
        # CATEGORY COMPARISON
        # -------------------------------------------------
        st.subheader(
            "Hybrid vs Naive — Category Comparison"
        )

        comparison_long = comparison_df.melt(
            id_vars=[
                "customer_state",
                "product_category",
            ],
            value_vars=[
                "hybrid_wape",
                "naive_wape",
                "holt_wape",
            ],
            var_name="Model",
            value_name="WAPE (%)",
        )

        comparison_long["Model"] = (
            comparison_long["Model"].map({
                "hybrid_wape": "Hybrid",
                "naive_wape": "Naive",
                "holt_wape": "Holt",
            })
        )

        comparison_long["State - Category"] = (
            comparison_long["customer_state"]
            + " - "
            + comparison_long["product_category"]
        )

        fig = px.bar(
            comparison_long,
            x="State - Category",
            y="WAPE (%)",
            color="Model",
            barmode="group",
            title="WAPE by State and Product Category",
            color_discrete_sequence=[
                "#2196F3",
                "#64B5F6",
                "#90A4AE",
            ],
        )

        fig.update_layout(
            height=550,
            xaxis_title="State - Product Category",
            yaxis_title="WAPE (%)",
            xaxis_tickangle=-60,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.divider()

        # -------------------------------------------------
        # OVERALL MODEL COMPARISON
        # -------------------------------------------------
        st.header(
            "Overall Forecasting Model Comparison"
        )

        st.markdown(
            "Final test-period comparison: January 2018 to August 2018."
        )

        performance = pd.DataFrame({
            "Model": [
                "Naive",
                "Holt",
                "LightGBM",
                "Hybrid",
            ],

            "MAE": [
                28.80,
                31.569,
                49.5682,
                30.052,
            ],

            "RMSE": [
                41.3831,
                45.2288,
                65.0791,
                43.284,
            ],

            "WAPE (%)": [
                18.9248,
                20.7443,
                32.5718,
                19.7475,
            ],
        })

        selected_metric = st.selectbox(
            "Select evaluation metric",
            [
                "WAPE (%)",
                "MAE",
                "RMSE",
            ],
        )

        sorted_performance = performance.sort_values(
            selected_metric,
            ascending=True,
        )

        fig = px.bar(
            sorted_performance,
            x="Model",
            y=selected_metric,
            color=selected_metric,
            text_auto=".2f",
            color_continuous_scale="Blues",
            title=f"Final Test Performance — {selected_metric}",
        )

        fig.update_layout(
            height=450,
            xaxis_title="Forecasting Model",
            yaxis_title=selected_metric,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        # Complete Metrics
        st.subheader(
            "Complete Test Performance Metrics"
        )

        st.dataframe(
            performance.round(3),
            width="stretch",
            hide_index=True,
        )

        best_model = performance.loc[
            performance["WAPE (%)"].idxmin()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Overall Model",
            best_model["Model"],
        )

        col2.metric(
            "Best Test WAPE",
            f"{best_model['WAPE (%)']:.2f}%",
        )

        col3.metric(
            "Hybrid Test WAPE",
            "19.75%",
        )

        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------
        st.subheader("Model Evaluation Insights")

        st.success(
            "The Naive model achieved the lowest overall "
            "test WAPE of 18.92%."
        )

        st.info(
            "The hybrid model uses validation performance to "
            "select Naive or Holt for each state-category "
            "combination. It improves several individual "
            "combinations but does not outperform Naive "
            "overall on the final test period."
        )

        st.warning(
            "LightGBM did not outperform the simpler forecasting "
            "baselines in this experiment. The available training "
            "history is relatively small."
        )

        # -------------------------------------------------
        # METHODOLOGY NOTES
        # -------------------------------------------------
        st.divider()

        st.subheader("Evaluation Methodology")

        st.markdown("""
- **Validation period:** April–December 2017.
- **Final test period:** January–August 2018.
- **Forecasting strategy:** Rolling one-step-ahead.
- **Naive:** Uses the previous month's demand.
- **Holt:** Uses an additive damped-trend model.
- **LightGBM:** Uses lagged demand and time-based features.
- **Hybrid:** Selects Naive or Holt using validation WAPE.

Lower MAE, RMSE and WAPE indicate better forecasting accuracy.
        """)

        st.caption(
            "Historical Olist demand data is used. The final test "
            "results represent historical evaluation, not guaranteed "
            "future forecasting performance."
        )


# =========================================================
# 10. FOOTER
# =========================================================
st.divider()

st.caption(
    "AI-Powered Predictive Inventory Pre-Positioning & "
    "Smart Hub Allocation System | Academic Project"
)

st.caption(
    "Historical data: Olist Brazilian E-Commerce Dataset. "
    "Inventory stock levels are hypothetical simulation inputs. "
    "Customer states represent delivery demand regions, "
    "not verified warehouse locations."
)        