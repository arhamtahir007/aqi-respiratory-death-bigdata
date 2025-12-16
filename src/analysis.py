# ============================================
# AQI & Respiratory Death Analysis (Big Data)
# Using Dask + Scikit-Learn
# ============================================

import dask.dataframe as dd
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from dask_ml.cluster import KMeans
from dask_ml.model_selection import train_test_split

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------
df = dd.read_csv(
    "data/aqi_respiratory.csv",
    assume_missing=True,
    dtype=str
)

# --------------------------------------------
# 2. CLEANING & TYPE FIXING
# --------------------------------------------

# Convert numeric columns
numeric_cols = [
    "Resp Death Rate", "Good Days", "Moderate Days",
    "Unhealthy for Sensitive Groups Days", "Unhealthy Days",
    "Very Unhealthy Days", "Hazardous Days", "Max AQI",
    "90th Percentile AQI", "Median AQI", "Days CO",
    "Days NO2", "Days Ozone", "Days PM2_5", "Days PM10"
]

for col in numeric_cols:
    df[col] = dd.to_numeric(df[col], errors="coerce")

df["year"] = dd.to_numeric(df["year"], errors="coerce")
df["fips"] = df["fips"].astype(str)

# --------------------------------------------
# 3. RENAME COLUMNS (STANDARDIZATION)
# --------------------------------------------
df = df.rename(columns={
    "Resp Death Rate": "resp_death_rate",
    "Good Days": "good_days",
    "Moderate Days": "moderate_days",
    "Unhealthy for Sensitive Groups Days": "sensitive_days",
    "Unhealthy Days": "unhealthy_days",
    "Very Unhealthy Days": "very_unhealthy_days",
    "Hazardous Days": "hazardous_days",
    "Median AQI": "median_aqi",
    "Max AQI": "max_aqi",
    "90th Percentile AQI": "p90_aqi",
    "Days CO": "co_days",
    "Days NO2": "no2_days",
    "Days Ozone": "ozone_days",
    "Days PM2_5": "pm25_days",
    "Days PM10": "pm10_days"
})

# --------------------------------------------
# 4. YEARLY TRENDS
# --------------------------------------------
yearly_trends = (
    df.groupby("year")
      .agg({
          "resp_death_rate": "mean",
          "median_aqi": "mean",
          "pm25_days": "mean",
          "ozone_days": "mean"
      })
      .compute()
)

yearly_trends.to_csv("outputs/yearly_trends.csv")

# --------------------------------------------
# 5. AQI CATEGORY DISTRIBUTION
# --------------------------------------------
aqi_categories = [
    "good_days",
    "moderate_days",
    "sensitive_days",
    "unhealthy_days",
    "very_unhealthy_days",
    "hazardous_days"
]

aqi_dist = df[aqi_categories].mean().compute().reset_index()
aqi_dist.columns = ["AQI_Category", "Average_Days"]
aqi_dist.to_csv("outputs/aqi_category_distribution.csv", index=False)

# --------------------------------------------
# 6. CORRELATION ANALYSIS
# --------------------------------------------
corr_features = [
    "resp_death_rate", "median_aqi", "max_aqi",
    "unhealthy_days", "pm25_days", "ozone_days",
    "no2_days", "co_days", "pm10_days"
]

corr_df = df[corr_features].sample(frac=0.25).compute()
corr_matrix = corr_df.corr()
corr_matrix.to_csv("outputs/correlation_matrix.csv")

# Convert correlation matrix to long format (Power BI friendly)
corr_long = corr_matrix.reset_index().melt(
    id_vars="index",
    var_name="Variable_Y",
    value_name="Correlation"
)

corr_long.rename(columns={"index": "Variable_X"}, inplace=True)

corr_long.to_csv("outputs/correlation_matrix_long.csv", index=False)


# --------------------------------------------
# 7. POLLUTANT IMPACT SUMMARY
# --------------------------------------------
pollutants = ["pm25_days", "ozone_days", "no2_days", "co_days", "pm10_days"]

pollutant_corr_df = pd.DataFrame({
    "Pollutant": pollutants,
    "Correlation_with_Resp_Death_Rate": [
        corr_matrix.loc["resp_death_rate", p] for p in pollutants
    ]
})

pollutant_corr_df.to_csv("outputs/pollutant_correlations.csv", index=False)

# --------------------------------------------
# 8. REGRESSION MODEL (SKLEARN)
# --------------------------------------------

# Select only rows with complete data
model_df = df[[
    "median_aqi",
    "pm25_days",
    "ozone_days",
    "unhealthy_days",
    "resp_death_rate"
]].dropna()

# Convert to pandas
model_df = model_df.compute()

X = model_df[[
    "median_aqi",
    "pm25_days",
    "ozone_days",
    "unhealthy_days"
]]

y = model_df["resp_death_rate"]

# Train/test split (sklearn version)
from sklearn.model_selection import train_test_split
X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
lin_model = LinearRegression()
lin_model.fit(X_train_pd, y_train_pd)

# Predict
predictions = lin_model.predict(X_test_pd)

# Save outputs
pred_df = pd.DataFrame({
    "Actual_Resp_Death_Rate": y_test_pd,
    "Predicted_Resp_Death_Rate": predictions
})
pred_df.to_csv("outputs/model_predictions.csv", index=False)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lin_model.coef_
})
coef_df.to_csv("outputs/regression_coefficients.csv", index=False)


coef_df.to_csv("outputs/regression_coefficients.csv", index=False)

# --------------------------------------------
# 9. FEATURE IMPORTANCE (RANDOM FOREST)
# --------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_pd, y_train_pd)

importance_df = pd.DataFrame({
    "Feature": X_train_pd.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("outputs/feature_importance.csv", index=False)

# --------------------------------------------
# 10. CLUSTERING (DASK-ML KMeans)
# --------------------------------------------
cluster_df = df[[
    "median_aqi",
    "pm25_days",
    "unhealthy_days",
    "resp_death_rate"
]].dropna()

# Convert to pandas for clustering
cluster_sample = cluster_df.sample(frac=0.3).compute()

kmeans = KMeans(n_clusters=3, random_state=42)

# Fit + predict
kmeans.fit(cluster_sample)
cluster_sample["Risk_Cluster"] = kmeans.predict(cluster_sample)

cluster_sample.to_csv("outputs/cluster_labels.csv", index=False)


# --------------------------------------------
# 11. COUNTY-LEVEL AGGREGATION
# --------------------------------------------
county_agg = (
    df.groupby("fips")
      .agg({
          "resp_death_rate": "mean",
          "median_aqi": "mean",
          "pm25_days": "mean",
          "ozone_days": "mean"
      })
      .compute()
)

county_agg.to_csv("outputs/county_level_aggregates.csv")

print("✅ ALL ANALYSES COMPLETE — FILES SAVED FOR POWER BI")
