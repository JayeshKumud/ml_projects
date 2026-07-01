# %%
# ============================================================
# BANGALORE HOME PRICE PREDICTION — CLEAN & PROFESSIONAL VERSION
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    ShuffleSplit,
    cross_val_score,
    GridSearchCV,
)
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import json

print("📌 Starting preprocessing pipeline...")

# %%
# ============================================================
# LOAD & INITIAL CLEANING
# ============================================================

print("➡ Loading dataset...")
df = pd.read_csv("../data/bengaluru_house_prices.csv")
print(f"✔ Loaded dataset with shape: {df.shape}")

# Drop unused columns
df = df.drop(["area_type", "society", "balcony", "availability"], axis=1)
print("✔ Dropped unused columns")

# Remove missing values
df = df.dropna()
print(f"✔ Removed missing values. New shape: {df.shape}")

# %%
# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("➡ Extracting BHK from size column...")
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))

print("➡ Converting total_sqft ranges to numeric...")


def convert_sqft(x):
    try:
        if "-" in x:
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None


df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df[df.total_sqft.notnull()]
print(f"✔ Cleaned total_sqft. Shape: {df.shape}")

# %%
# Show df4.head() as requested
print("📌 Displaying cleaned sqft DataFrame (df4 equivalent):")
df4 = df.copy()
df4.head()

# %%
# ============================================================
# PRICE PER SQFT
# ============================================================

print("➡ Creating price_per_sqft feature...")
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]
print("✔ price_per_sqft created")

# %%
# ============================================================
# LOCATION CLEANING & DIMENSIONALITY REDUCTION
# ============================================================

print("➡ Cleaning location values...")
df["location"] = df["location"].str.strip()

location_counts = df["location"].value_counts()
rare_locations = location_counts[location_counts <= 10].index

df["location"] = df["location"].apply(lambda x: "other" if x in rare_locations else x)
print("✔ Location dimensionality reduced")

# %%
# ============================================================
# OUTLIER REMOVAL
# ============================================================

print("➡ Removing sqft-per-BHK outliers...")
df = df[df.total_sqft / df.bhk >= 300]
print(f"✔ After sqft-per-BHK rule: {df.shape}")

print("➡ Removing price_per_sqft outliers...")


def remove_pps_outliers(df):
    cleaned = []
    for loc, subdf in df.groupby("location"):
        mean, std = subdf.price_per_sqft.mean(), subdf.price_per_sqft.std()
        filtered = subdf[
            (subdf.price_per_sqft > mean - std) & (subdf.price_per_sqft <= mean + std)
        ]
        cleaned.append(filtered)
    return pd.concat(cleaned, ignore_index=True)


df = remove_pps_outliers(df)
print(f"✔ After price_per_sqft outlier removal: {df.shape}")

print("➡ Removing BHK outliers...")


def remove_bhk_outliers(df):
    indices_to_remove = []
    for loc, loc_df in df.groupby("location"):
        bhk_stats = {
            bhk: {
                "mean": subdf.price_per_sqft.mean(),
                "std": subdf.price_per_sqft.std(),
                "count": subdf.shape[0],
            }
            for bhk, subdf in loc_df.groupby("bhk")
        }
        for bhk, subdf in loc_df.groupby("bhk"):
            prev_stats = bhk_stats.get(bhk - 1)
            if prev_stats and prev_stats["count"] > 5:
                bad_idx = subdf[subdf.price_per_sqft < prev_stats["mean"]].index
                indices_to_remove.extend(bad_idx)
    return df.drop(indices_to_remove)


df = remove_bhk_outliers(df)
print(f"✔ After BHK outlier removal: {df.shape}")

print("➡ Applying bathroom rule...")
df = df[df.bath < df.bhk + 2]
print(f"✔ After bathroom rule: {df.shape}")

# %%
# ============================================================
# FINAL CLEANUP & ENCODING
# ============================================================

df = df.drop(["size", "price_per_sqft"], axis=1)
print("✔ Dropped size & price_per_sqft")

df = pd.get_dummies(df, columns=["location"], drop_first=True)
print(f"✔ One-hot encoded locations. Final shape: {df.shape}")

# %%
# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)
print("✔ Train-test split completed")

# %%
# ============================================================
# TRAIN LINEAR REGRESSION MODEL
# ============================================================

print("➡ Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print(f"✔ Linear Regression R² Score: {lr_model.score(X_test, y_test):.4f}")

# %%
# ============================================================
# CROSS VALIDATION
# ============================================================

print("➡ Running cross-validation...")
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv)
print(f"✔ Cross-validation scores: {cv_scores}")

# %%
# ============================================================
# GRIDSEARCHCV — BEST MODEL SELECTION
# ============================================================

print("➡ Running GridSearchCV for best model...")


def find_best_model(X, y):
    models = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {
                "fit_intercept": [True, False],
                "copy_X": [True, False],
                "positive": [True, False],
            },
        },
        "lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [0.1, 1, 5],
                "selection": ["random", "cyclic"],
            },
        },
        "decision_tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "criterion": ["squared_error", "friedman_mse"],
                "splitter": ["best", "random"],
            },
        },
    }

    results = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for name, cfg in models.items():
        gs = GridSearchCV(cfg["model"], cfg["params"], cv=cv, return_train_score=False)
        gs.fit(X, y)
        results.append(
            {
                "model": name,
                "best_score": gs.best_score_,
                "best_params": gs.best_params_,
            }
        )
        print(f"✔ {name} best score: {gs.best_score_}, params: {gs.best_params_}")

    return pd.DataFrame(results)


best_models_df = find_best_model(X, y)
print(best_models_df)

# %%
# ============================================================
# PRICE PREDICTION FUNCTION
# ============================================================


def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(X.columns))
    x[0], x[1], x[2] = sqft, bath, bhk

    loc_col = f"location_{location.lower()}"
    if loc_col in X.columns:
        x[X.columns.get_loc(loc_col)] = 1

    return lr_model.predict([x])[0]


print("➡ Sample predictions:")
print("JP Nagar:", predict_price("1st Phase JP Nagar", 1000, 2, 2))
print("Indira Nagar:", predict_price("Indira Nagar", 1000, 3, 3))

# %%
# ============================================================
# EXPORT MODEL & METADATA
# ============================================================

print("➡ Exporting model and metadata...")

with open("../server/artifacts/home_prices_model.pickle", "wb") as f:
    pickle.dump(lr_model, f)

columns = {"data_columns": [col.lower() for col in X.columns]}
with open("../server/artifacts/columns.json", "w") as f:
    json.dump(columns, f)

print("✔ Model and columns exported successfully")
