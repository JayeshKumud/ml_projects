import os
import json
import pickle
import numpy as np

# ------------------------------------------------------------
# GLOBAL ARTIFACTS
# ------------------------------------------------------------
_data_columns = None
_locations = None
_model = None

# Resolve absolute path to artifacts directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# ------------------------------------------------------------
# LOAD ARTIFACTS
# ------------------------------------------------------------
def load_saved_artifacts():
    """
    Load model and metadata (columns.json) from artifacts directory.
    Uses absolute paths to avoid FileNotFound errors.
    """
    global _data_columns, _locations, _model

    print("📌 Loading saved artifacts...")

    # Load column metadata
    columns_path = os.path.join(ARTIFACTS_DIR, "columns.json")
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"❌ columns.json not found at: {columns_path}")

    with open(columns_path, "r") as f:
        _data_columns = json.load(f)["data_columns"]

    # First 3 columns are numeric: sqft, bath, bhk
    _locations = _data_columns[3:]
    print(f"✔ Loaded {_data_columns.__len__()} data columns")
    print(f"✔ Loaded {len(_locations)} location entries")

    # Load trained model
    model_path = os.path.join(ARTIFACTS_DIR, "home_prices_model.pickle")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    print("✔ Model loaded successfully")
    print("📌 Artifacts loading complete")


# ------------------------------------------------------------
# PRICE PREDICTION
# ------------------------------------------------------------
def get_estimated_price(location: str, sqft: float, bhk: int, bath: int) -> float:
    """
    Predict home price using trained model.
    """
    if _model is None or _data_columns is None:
        raise RuntimeError("Artifacts not loaded. Call load_saved_artifacts() first.")

    location = location.lower()

    # Find location index (if exists)
    try:
        loc_index = _data_columns.index(location)
    except ValueError:
        loc_index = -1  # location not found → treat as "other"

    # Build feature vector
    x = np.zeros(len(_data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = _model.predict([x])[0]
    return round(predicted_price, 2)


# ------------------------------------------------------------
# GETTERS
# ------------------------------------------------------------
def get_location_names():
    """Return list of available locations."""
    return _locations


def get_data_columns():
    """Return full list of model input columns."""
    return _data_columns


# ------------------------------------------------------------
# DEBUG / LOCAL TESTING
# ------------------------------------------------------------
if __name__ == "__main__":
    load_saved_artifacts()

    print("Available locations:", get_location_names())
    print("Test Predictions:")
    print(get_estimated_price("1st Phase JP Nagar", 1000, 3, 3))
    print(get_estimated_price("1st Phase JP Nagar", 1000, 2, 2))
    print(get_estimated_price("Kalhalli", 1000, 2, 2))  # other location
    print(get_estimated_price("Ejipura", 1000, 2, 2))  # other location
