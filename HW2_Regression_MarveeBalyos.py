"""
CS 577 – Principles and Techniques of Data Science
Homework 2: Regression

Student: Marvee Balyos
RedID: 131758247
Date: 9/16/25

"""

# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Setup ----------
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_ohe(drop_first=True):
    """Compatibility helper for scikit-learn version differences"""
    try:
        return OneHotEncoder(drop="first" if drop_first else None,
                             handle_unknown="ignore",
                             sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first" if drop_first else None,
                             handle_unknown="ignore",
                             sparse=False)

# ==========================================================
# PART A: Simple Linear Regression
# ==========================================================
df_simple = pd.read_csv("data_linear_regression.csv", sep="\t")
df_simple.columns = df_simple.columns.str.strip()
print("DEBUG Part A columns:", df_simple.columns)

X = df_simple[["YearsExperience"]].values
y = df_simple["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

lin_reg = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print("=== Part A: Simple Linear Regression ===")
print("R^2:", r2_score(y_test, y_pred))
print("RMSE:", rmse(y_test, y_pred))

plt.figure()
plt.scatter(X_train, y_train, label="Training Data")
xs = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
plt.plot(xs, lin_reg.predict(xs), label="Regression Line")
plt.title("Simple Linear Regression (Train)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.savefig(OUT / "partA_train.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.scatter(X_test, y_test, label="Test Data")
xs_all = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.plot(xs_all, lin_reg.predict(xs_all), label="Regression Line")
plt.title("Simple Linear Regression (Test)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.savefig(OUT / "partA_test.png", bbox_inches="tight")
plt.close()


# ==========================================================
# PART B: Multiple Linear Regression
# ==========================================================
# ✅ Correct fix: file is tab-separated
df_multi = pd.read_csv("data_multiple_regression.csv", sep="\t")
df_multi.columns = df_multi.columns.str.strip()
print("DEBUG Part B columns:", df_multi.columns)
print("DEBUG Part B head:\n", df_multi.head())

X_multi = df_multi.drop(columns=["Profit"])
y_multi = df_multi["Profit"].values

categorical_features = ["State"]
ct = ColumnTransformer(
    [("onehot", make_ohe(drop_first=True), categorical_features)],
    remainder="passthrough"
)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

multi_model = Pipeline([("prep", ct), ("reg", LinearRegression())])
multi_model.fit(X_train_m, y_train_m)
y_pred_m = multi_model.predict(X_test_m)

print("\n=== Part B: Multiple Linear Regression ===")
print("R^2:", r2_score(y_test_m, y_pred_m))
print("RMSE:", rmse(y_test_m, y_pred_m))

plt.figure()
plt.scatter(y_test_m, y_pred_m)
m = min(y_test_m.min(), y_pred_m.min())
M = max(y_test_m.max(), y_pred_m.max())
plt.plot([m, M], [m, M])
plt.title("Multiple Regression (Predicted vs Actual)")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.savefig(OUT / "partB_pred_vs_actual.png", bbox_inches="tight")
plt.close()


# ==========================================================
# PART C: Polynomial Regression
# ==========================================================
df_poly = pd.read_csv("data_polynomial_regression.csv", sep="\t")
df_poly.columns = df_poly.columns.str.strip()
print("DEBUG Part C columns:", df_poly.columns)

Xp = df_poly[["Level"]].values.astype(float)
yp = df_poly["Salary"].values.astype(float)

lin_reg_poly_baseline = LinearRegression().fit(Xp, yp)

DEGREE = 4
poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
Xp_poly = poly.fit_transform(Xp)
lin_reg_poly = LinearRegression().fit(Xp_poly, yp)

plt.figure()
plt.scatter(Xp, yp, label="Data")
xg = np.linspace(Xp.min(), Xp.max(), 200).reshape(-1, 1)
plt.plot(xg, lin_reg_poly_baseline.predict(xg), label="Linear Fit")
plt.title("Polynomial Dataset - Linear Fit")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.savefig(OUT / "partC_linear_fit.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.scatter(Xp, yp, label="Data")
plt.plot(xg, lin_reg_poly.predict(poly.transform(xg)),
         label=f"Polynomial Fit (degree={DEGREE})")
plt.title(f"Polynomial Regression (degree={DEGREE})")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.savefig(OUT / "partC_poly_fit.png", bbox_inches="tight")
plt.close()

new_level = np.array([[6.5]])
pred_lin = lin_reg_poly_baseline.predict(new_level)[0]
pred_poly = lin_reg_poly.predict(poly.transform(new_level))[0]

print("\n=== Part C: Polynomial Regression ===")
print("Prediction for Level 6.5 (Linear):", pred_lin)
print("Prediction for Level 6.5 (Polynomial):", pred_poly)

print(f"\nAll plots have been saved to: {OUT.resolve()}")