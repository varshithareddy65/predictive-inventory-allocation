# Predictive Inventory Pre-Positioning & Smart Hub Allocation System

A machine learning-based inventory planning project that combines demand forecasting, safety stock estimation and inventory allocation using the Olist Brazilian E-Commerce dataset.

## 1. Project Overview

Efficient inventory planning is essential for e-commerce businesses to reduce stock shortages and improve product availability.

This project develops a predictive inventory planning system that analyzes historical e-commerce demand, forecasts future demand across geographical regions and product categories, and generates inventory allocation recommendations under simulated stock constraints.

The system uses multiple forecasting approaches, including statistical baselines and LightGBM, followed by two inventory allocation strategies.

An interactive Streamlit dashboard presents demand patterns, forecasting results and allocation recommendations.

## 2. Problem Statement

E-commerce businesses need to anticipate regional product demand and distribute limited inventory efficiently.

This project addresses three main problems:

- How can historical order data be used to forecast future demand?
- How can additional safety stock account for demand uncertainty?
- How can limited inventory be distributed across geographical demand regions and product categories?

## 3. Dataset

**Dataset:** Olist Brazilian E-Commerce Public Dataset.

The dataset contains historical e-commerce information, including:

| Dataset | Description |
|---|---|
| Customers | Customer information and geographical locations |
| Orders | Order dates and order status |
| Order Items | Products purchased and seller information |
| Products | Product details and categories |
| Sellers | Seller information and locations |
| Geolocation | Geographical coordinates |
| Payments | Payment information |
| Reviews | Customer review information |

The project uses customer-state and product-category combinations to analyze historical demand.

**Dataset limitation:** The Olist dataset does not provide actual warehouse inventory, hub capacities, stockout records or search-trend data. Therefore, inventory availability and allocation are simulated rather than based on actual warehouse operations.

The original dataset is not included in this repository because of its size.

## 4. Data Processing

The preprocessing workflow includes:

1. Loading and inspecting the original Olist datasets.
2. Processing order and order-item records.
3. Combining geographical and product-category information.
4. Preparing state-category demand records.
5. Generating monthly demand datasets.
6. Incorporating order-status information.
7. Preparing forecasting experiment datasets.

The processed datasets are generated as Parquet files for efficient data processing.

## 5. Feature Engineering

Historical demand features are generated for forecasting.

| Feature | Description |
|---|---|
| Lag 1 | Demand in the previous month |
| Lag 2 | Demand two months earlier |
| Lag 3 | Demand three months earlier |
| Lag 6 | Demand six months earlier |
| Rolling Mean 3 | Three-period average demand |
| Rolling Mean 6 | Six-period average demand |
| Rolling Std 3 | Three-period demand variability |

Additional categorical and temporal features include customer state, product category and month.

## 6. Demand Forecasting

Multiple forecasting approaches were implemented and evaluated.

- Naive Forecasting
- Moving Average
- Seasonal Naive
- Holt's Exponential Smoothing
- LightGBM
- Hybrid Forecasting

LightGBM was trained using 10 features to capture historical demand patterns across state-category combinations.

The forecasting experiment evaluates the top 20 selected state-category combinations.

## 7. Model Evaluation

The forecasting approaches were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and Weighted Absolute Percentage Error (WAPE).

### Forecasting Results

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 28.80 | 41.38 | 18.92% |
| Moving Average | — | — | 19.95% |
| Seasonal Naive | — | — | 62.80% |
| Holt | — | — | 20.74% |
| LightGBM | — | — | 32.57% |
| Hybrid | — | — | 19.75% |

The Naive forecasting baseline achieved the best overall WAPE in this experiment.

The hybrid approach selected Holt forecasting for 14 combinations and Naive forecasting for 6 combinations. However, it did not outperform the Naive baseline overall.

## 8. Safety Stock Estimation

Safety stock is additional inventory maintained to protect against uncertainty in future demand.

The project uses validation forecasting errors to estimate safety stock.

The implementation uses the standard deviation of validation errors and a Z-score of 1.28 as a heuristic safety-stock calculation.

The resulting safety-stock estimates are combined with forecast demand to calculate inventory requirements.

## 9. Inventory Allocation

Two inventory allocation approaches were implemented.

### V1 – Proportional Inventory Allocation

The first approach distributes available category-level inventory proportionally according to the required stock of each state-category combination.

The available inventory is simulated as 80% of the total calculated requirement.

The Largest Remainder Method is used to convert proportional allocations into integer inventory units while maintaining the total available stock constraint.

### V2 – Improved Inventory Allocation

The second approach introduces an alternative allocation strategy to reduce simulated unmet forecast demand.

### Allocation Comparison

| Metric | V1 | V2 |
|---|---:|---:|
| Simulated unmet forecast demand | 125.64 | 116.92 |

V2 achieved an approximately 6.94% reduction in simulated unmet forecast demand compared with V1 under the same simulated stock constraint.

This comparison represents a simulation result, not a measurement of actual warehouse performance.

## 10. Interactive Dashboard

A Streamlit dashboard provides an interactive view of the project results.

The dashboard contains six sections:

1. Executive Overview
2. Demand Analysis
3. Geographical Demand
4. Demand Forecasting
5. Inventory Allocation
6. Model Performance

It presents demand trends, geographical patterns, forecasting comparisons and allocation recommendations.

## 11. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| DuckDB | Data querying and processing |
| PyArrow | Parquet processing |
| Scikit-learn | Evaluation metrics |
| LightGBM | Demand forecasting |
| Statsmodels | Holt forecasting |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |

## 12. Project Structure

```text
predictive-inventory-allocation/
│
├── app.py
│
├── datasets/                 # Generated locally; not uploaded
│   └── olist/                # Original Olist CSV files
│
├── models/
│   └── demand_forecasting_results.json
│
├── src/
│   ├── data_preparation/
│   ├── feature_engineering/
│   ├── models/
│   └── validation/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 13. Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Olist Brazilian E-Commerce dataset

### Step 1: Clone the repository

```bash
git clone https://github.com/varshithareddy65/predictive-inventory-allocation.git

cd predictive-inventory-allocation
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add the dataset

Place the original Olist CSV files inside:

```text
datasets/olist/
```

### Step 4: Generate processed datasets

Run the data preparation and feature engineering scripts from the project root.

The scripts are organized into data preparation, feature engineering, model training and validation stages. Required intermediate datasets must be generated before running dependent scripts.

### Step 5: Train the forecasting model

```bash
python src/models/train_demand_forecasting.py
```

The trained model is saved locally under `models/`.

### Step 6: Run the dashboard

```bash
streamlit run app.py
```

**Note:** The processed datasets and trained LightGBM model are excluded from the GitHub repository. They must be generated locally before running the complete dashboard.

## 14. Key Learning Outcomes

- Processing real-world e-commerce datasets.
- Analyzing geographical and product-category demand.
- Developing historical demand forecasting features.
- Comparing statistical forecasting and machine learning approaches.
- Evaluating forecasting models using MAE, RMSE and WAPE.
- Estimating safety stock using validation errors.
- Implementing proportional inventory allocation.
- Comparing alternative allocation strategies.
- Developing interactive dashboards using Streamlit.

## 15. Future Improvements

- Incorporate actual inventory and warehouse capacity data.
- Integrate real-time demand and search-trend information.
- Explore optimization algorithms for inventory distribution.
- Incorporate transportation and delivery costs.
- Experiment with additional forecasting models.
- Develop a more advanced inventory planning interface.

## 16. Project Summary

This project combines historical e-commerce demand analysis, machine learning-based forecasting, safety stock estimation and inventory allocation into an integrated inventory planning workflow.

It demonstrates how predictive analytics can support inventory planning decisions under simulated resource constraints.