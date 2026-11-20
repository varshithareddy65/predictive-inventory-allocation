
import lightgbm as lgb
import pandas as pd

model = lgb.Booster(
    model_file="models/lightgbm_purchase_model.txt"
)

importance = pd.DataFrame({
    "feature": model.feature_name(),
    "gain": model.feature_importance(
        importance_type="gain"
    ),
    "split": model.feature_importance(
        importance_type="split"
    )
})

importance["gain_percent"] = (
    importance["gain"] /
    importance["gain"].sum() * 100
)

importance = importance.sort_values(
    "gain_percent",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nSaved: models/feature_importance.csv")
