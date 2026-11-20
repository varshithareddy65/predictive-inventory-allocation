
import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = "datasets/demand_forecasting_features.parquet"
OUTPUT = "datasets/lightgbm_demand_predictions.parquet"
MODEL_PATH = "models/lightgbm_demand_forecasting.txt"
RESULT_PATH = "models/demand_forecasting_results.json"

Path("models").mkdir(exist_ok=True)

con = duckdb.connect()
df = con.execute(
    f"SELECT * FROM read_parquet('{INPUT}')"
).fetchdf()
con.close()

df["order_month"] = pd.to_datetime(df["order_month"])

features = [
    "customer_state",
    "product_category",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
]

categorical_features = [
    "customer_state",
    "product_category",
]

for col in categorical_features:
    df[col] = df[col].astype("category")

train = df[df["order_month"] < "2018-01-01"].copy()
test = df[df["order_month"] >= "2018-01-01"].copy()

X_train = train[features]
y_train = train["demand"]

X_test = test[features]
y_test = test["demand"]

print("Training rows:", len(train))
print("Testing rows:", len(test))

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=100,
    learning_rate=0.03,
    num_leaves=7,
    max_depth=3,
    min_child_samples=10,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features
)

predictions = np.maximum(
    0,
    model.predict(X_test)
)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

total_actual = y_test.sum()
wape = (
    np.abs(y_test.to_numpy() - predictions).sum()
    / total_actual * 100
)

test_results = test[
    ["order_month", "customer_state", "product_category", "demand"]
].copy()

test_results["prediction"] = predictions

test_results.to_parquet(
    OUTPUT,
    index=False,
    compression="zstd"
)

model.booster_.save_model(MODEL_PATH)

results = {
    "model": "LightGBM",
    "training_rows": len(train),
    "testing_rows": len(test),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "WAPE_percent": float(wape),
    "total_actual": float(total_actual),
}

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nLIGHTGBM FORECASTING RESULTS")
for key, value in results.items():
    print(f"{key}: {value}")

print("\nFeature importance:")
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(importance.to_string(index=False))

print("\nSaved model:", MODEL_PATH)
print("Saved predictions:", OUTPUT)
print("Saved metrics:", RESULT_PATH)
