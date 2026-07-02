"""
PRODIGY_ML_01 - House Price Prediction using Linear Regression
Prodigy Infotech Machine Learning Internship

Task: Implement a linear regression model to predict house prices based on
      square footage, number of bedrooms, and number of bathrooms.

Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Instructions:
    1. Download train.csv from the Kaggle link above
    2. Place it in the same directory as this script
    3. Run: python task01_house_price_prediction.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  PRODIGY_ML_01 — House Price Prediction")
print("=" * 60)

try:
    df = pd.read_csv("train.csv")
    print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("\n⚠️  train.csv not found. Generating synthetic dataset for demo...")
    np.random.seed(42)
    n = 1000
    sqft     = np.random.randint(500, 5000, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 5, n)
    price = (sqft * 120 + bedrooms * 8000 + bathrooms * 5000
             + np.random.normal(0, 15000, n))
    df = pd.DataFrame({
        "GrLivArea": sqft,
        "BedroomAbvGr": bedrooms,
        "FullBath": bathrooms,
        "HalfBath": 0,
        "SalePrice": price.clip(50000)
    })
    print(f"   Synthetic dataset generated: {df.shape[0]} samples")

# ─────────────────────────────────────────────
# 2. SELECT & ENGINEER FEATURES
# ─────────────────────────────────────────────
# Core features required by the task
REQUIRED_COLS = ["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]
OPTIONAL_COLS = ["HalfBath", "TotRmsAbvGrd", "GarageArea",
                 "TotalBsmtSF", "OverallQual", "YearBuilt"]

available = [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in df.columns]
data = df[available].dropna().copy()

# Combine bathrooms: FullBath + 0.5 * HalfBath
if "HalfBath" in data.columns:
    data["TotalBathrooms"] = data["FullBath"] + 0.5 * data["HalfBath"]
else:
    data["TotalBathrooms"] = data["FullBath"]

# Feature list for modelling
feature_cols = ["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]
for col in ["TotRmsAbvGrd", "GarageArea", "TotalBsmtSF", "OverallQual", "YearBuilt"]:
    if col in data.columns:
        feature_cols.append(col)

X = data[feature_cols]
y = data["SalePrice"]

print(f"\n📊 Features used : {feature_cols}")
print(f"   Samples       : {len(X)}")
print(f"   Target range  : ${y.min():,.0f}  –  ${y.max():,.0f}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Task-01 | EDA – House Price Relationships", fontsize=14, fontweight="bold")

axes[0].scatter(data["GrLivArea"], y, alpha=0.4, color="#4C72B0", s=10)
axes[0].set_xlabel("Living Area (sq ft)")
axes[0].set_ylabel("Sale Price ($)")
axes[0].set_title("Price vs Square Footage")

axes[1].boxplot([y[data["BedroomAbvGr"] == b] for b in sorted(data["BedroomAbvGr"].unique())],
                labels=sorted(data["BedroomAbvGr"].unique()))
axes[1].set_xlabel("Number of Bedrooms")
axes[1].set_ylabel("Sale Price ($)")
axes[1].set_title("Price vs Bedrooms")

axes[2].scatter(data["TotalBathrooms"], y, alpha=0.4, color="#DD8452", s=15)
axes[2].set_xlabel("Total Bathrooms")
axes[2].set_ylabel("Sale Price ($)")
axes[2].set_title("Price vs Bathrooms")

plt.tight_layout()
plt.savefig("task01_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📈 EDA plot saved → task01_eda.png")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

y_pred      = model.predict(X_test_sc)
y_pred_train = model.predict(X_train_sc)

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)
r2_tr = r2_score(y_train, y_pred_train)

cv_scores = cross_val_score(LinearRegression(), X_train_sc, y_train,
                             cv=5, scoring="r2")

print("\n" + "─" * 40)
print("  MODEL PERFORMANCE")
print("─" * 40)
print(f"  RMSE             : ${rmse:>12,.2f}")
print(f"  MAE              : ${mae:>12,.2f}")
print(f"  R²  (test)       : {r2:.4f}")
print(f"  R²  (train)      : {r2_tr:.4f}")
print(f"  CV R² (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("─" * 40)

print("\n📌 Feature Coefficients:")
coef_df = pd.DataFrame({"Feature": feature_cols,
                         "Coefficient": model.coef_}).sort_values("Coefficient", ascending=False)
print(coef_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. RESULT PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Task-01 | Linear Regression — Results", fontsize=14, fontweight="bold")

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.4, color="#4C72B0", s=15)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect prediction")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title(f"Actual vs Predicted  (R² = {r2:.3f})")
axes[0].legend()

# Residuals
residuals = y_test - y_pred
axes[1].hist(residuals, bins=40, color="#55A868", edgecolor="white")
axes[1].axvline(0, color="red", lw=2, linestyle="--")
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("task01_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 Results plot saved → task01_results.png")

# ─────────────────────────────────────────────
# 8. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("\n🏠 Sample Predictions:")
sample_data = pd.DataFrame({
    "GrLivArea":     [1200, 2000, 3500],
    "BedroomAbvGr":  [2,    3,    4],
    "TotalBathrooms":[1,    2,    3],
})
# Fill extra features with column means if present
for col in feature_cols:
    if col not in sample_data.columns:
        sample_data[col] = X[col].mean()

sample_sc = scaler.transform(sample_data[feature_cols])
preds = model.predict(sample_sc)
for i, (_, row) in enumerate(sample_data.iterrows()):
    print(f"  {int(row.GrLivArea)} sqft, "
          f"{int(row.BedroomAbvGr)} bed, "
          f"{row.TotalBathrooms:.1f} bath  →  ${preds[i]:,.0f}")

print("\n✅ Task-01 complete!\n")
