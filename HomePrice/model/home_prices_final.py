# %%
# ------------------------------------------------------------
# MODEL & PROJECT DOCUMENTATION — BANGALORE HOME PRICE PREDICTION
# ------------------------------------------------------------
# PROJECT GOAL:
#   Build a regression model to predict home prices in Bangalore
#   using cleaned, engineered, and encoded real-estate features.
#
# MODEL USED:
#   Linear Regression (final model)
#
# WHY LINEAR REGRESSION?
#   • Simple and interpretable
#   • Works well when relationship between features and target
#     is approximately linear
#   • Performs well after removing outliers
#
# OTHER MODELS TESTED:
#   • Lasso Regression (L1 Regularization)
#   • Decision Tree Regression
#   • GridSearchCV used to find best model
#
# FEATURES USED:
#   • total_sqft
#   • bath
#   • bhk
#   • location (one-hot encoded)
#
# WHAT THIS CODE DOES (STEP-BY-STEP):
#   1. Load dataset
#   2. Drop unnecessary columns
#   3. Handle missing values
#   4. Feature engineering (bhk, total_sqft conversion)
#   5. Create price_per_sqft
#   6. Dimensionality reduction for location
#   7. Remove outliers using business logic
#   8. Remove outliers using standard deviation
#   9. Remove outliers using bhk comparison
#  10. Encode location using one-hot encoding
#  11. Train Linear Regression model
#  12. Evaluate using cross-validation
#  13. Use GridSearchCV to find best model
#  14. Predict price for sample properties
#  15. Export model + columns to files
#
# PRINTED OUTPUTS:
#   • DataFrame heads
#   • Shapes
#   • Scores
#   • Predictions
#
# ------------------------------------------------------------


# %%
# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20, 10)

# %%
# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
df1 = pd.read_csv("../data/bengaluru_house_prices.csv")
df1.head()  # Important DataFrame

df1.shape
df1.columns
df1["area_type"].value_counts()

# %%
# ------------------------------------------------------------
# DROP UNUSED COLUMNS
# ------------------------------------------------------------
df2 = df1.drop(["area_type", "society", "balcony", "availability"], axis="columns")
df2.head()  # Important DataFrame

# %%
# ------------------------------------------------------------
# HANDLE MISSING VALUES
# ------------------------------------------------------------
df3 = df2.dropna()
df3.head()  # Important DataFrame

# %%
# ------------------------------------------------------------
# FEATURE ENGINEERING — CREATE BHK
# ------------------------------------------------------------
df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))
df3.bhk.unique()


# %%
# ------------------------------------------------------------
# CLEAN total_sqft (convert ranges → average)
# ------------------------------------------------------------
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False


df3[~df3["total_sqft"].apply(is_float)].head(10)


def convert_sqft_to_num(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)

# %%
# ------------------------------------------------------------
# FEATURE ENGINEERING — price_per_sqft
# ------------------------------------------------------------
df5 = df4.copy()
df5["price_per_sqft"] = df5["price"] * 100000 / df5["total_sqft"]
df5.head()  # Important DataFrame

df5_stats = df5["price_per_sqft"].describe()

# %%
# ------------------------------------------------------------
# DIMENSIONALITY REDUCTION — LOCATION
# ------------------------------------------------------------
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5["location"].value_counts()

location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(
    lambda x: "other" if x in location_stats_less_than_10 else x
)
df5.head(10)

# %%
# ------------------------------------------------------------
# OUTLIER REMOVAL — sqft per bhk rule
# ------------------------------------------------------------
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
df6.shape


# %%
# ------------------------------------------------------------
# OUTLIER REMOVAL — price_per_sqft using mean & std
# ------------------------------------------------------------
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[
            (subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df7 = remove_pps_outliers(df6)
df7.shape


# %%
# ------------------------------------------------------------
# OUTLIER REMOVAL — bhk comparison logic
# ------------------------------------------------------------
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0],
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < stats["mean"]].index.values,
                )
    return df.drop(exclude_indices, axis="index")


df8 = remove_bhk_outliers(df7)
df8.shape

# %%
# ------------------------------------------------------------
# OUTLIER REMOVAL — bathroom rule
# ------------------------------------------------------------
df9 = df8[df8.bath < df8.bhk + 2]
df9.shape

df10 = df9.drop(["size", "price_per_sqft"], axis="columns")
df10.head(3)

# %%
# ------------------------------------------------------------
# ONE-HOT ENCODING — LOCATION
# ------------------------------------------------------------
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop("other", axis="columns")], axis="columns")
df12 = df11.drop("location", axis="columns")
df12.head(2)

# %%
# ------------------------------------------------------------
# TRAIN LINEAR REGRESSION MODEL
# ------------------------------------------------------------
X = df12.drop(["price"], axis="columns")
y = df12.price

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

from sklearn.linear_model import LinearRegression

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

print("Linear Regression Test Score:", lr_clf.score(X_test, y_test))

# %%
# ------------------------------------------------------------
# CROSS VALIDATION SCORE
# ------------------------------------------------------------
from sklearn.model_selection import ShuffleSplit, cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

# %%
# ------------------------------------------------------------
# GRIDSEARCHCV — FIND BEST MODEL
# ------------------------------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


def find_best_model_using_gridsearchcv(X, y):
    algos = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {"normalize": [True, False]},
        },
        "lasso": {
            "model": Lasso(),
            "params": {"alpha": [1, 2], "selection": ["random", "cyclic"]},
        },
        "decision_tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "criterion": ["mse", "friedman_mse"],
                "splitter": ["best", "random"],
            },
        },
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for algo_name, config in algos.items():
        gs = GridSearchCV(
            config["model"], config["params"], cv=cv, return_train_score=False
        )
        gs.fit(X, y)
        scores.append(
            {
                "model": algo_name,
                "best_score": gs.best_score_,
                "best_params": gs.best_params_,
            }
        )

    return pd.DataFrame(scores, columns=["model", "best_score", "best_params"])


find_best_model_using_gridsearchcv(X, y)


# %%
# ------------------------------------------------------------
# PRICE PREDICTION FUNCTION
# ------------------------------------------------------------
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


print(predict_price("1st Phase JP Nagar", 1000, 2, 2))
print(predict_price("Indira Nagar", 1000, 3, 3))

# %%
# ------------------------------------------------------------
# EXPORT MODEL + COLUMNS
# ------------------------------------------------------------
import pickle

with open("banglore_home_prices_model.pickle", "wb") as f:
    pickle.dump(lr_clf, f)

import json

columns = {"data_columns": [col.lower() for col in X.columns]}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
