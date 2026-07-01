# %%
# ------------------------------------------------------------
# FLASK API — PURPOSE & DESCRIPTION
# ------------------------------------------------------------
# WHY WE USE FLASK:
# Flask is a lightweight Python web framework used to expose machine‑learning
# models as HTTP APIs. ML models normally run inside Python scripts or notebooks,
# but real applications (web apps, mobile apps, dashboards) cannot execute Python
# directly. They need an API endpoint that they can call over HTTP.
#
# WHAT FLASK DOES IN THIS PROJECT:
#   • Loads the trained home‑price prediction model and metadata
#   • Provides REST API endpoints for:
#       - Fetching available locations
#       - Predicting home prices
#   • Acts as a backend server that communicates with your frontend UI
#   • Converts Python function outputs into JSON responses
#
# WHY FLASK IS NEEDED:
#   • Enables model deployment without heavy frameworks
#   • Allows any client (HTML/JS, React, Android, Postman) to call your model
#   • Handles GET/POST requests cleanly
#   • Provides a simple, production‑ready API layer
#
# SUMMARY:
# Flask is the bridge between your trained ML model and the user interface.
# ------------------------------------------------------------


# %% Flask API for Home Price Prediction

from flask import Flask, request, jsonify
import util  # custom module containing ML model loading & prediction

# ------------------------------------------------------------
# INITIALIZE FLASK APPLICATION
# ------------------------------------------------------------
# Flask(__name__) creates the application instance.
# It registers the current Python module as the app context.
# Flask uses __name__ to:
#   • locate resources
#   • map routes to functions
#   • enable debugging and auto‑reload
#
# 'app' becomes the central web server object.
app = Flask(__name__)


# ------------------------------------------------------------
# API 1: Return all location names from saved artifacts
# ------------------------------------------------------------
@app.route("/get_location_names", methods=["GET"])
def get_location_names():
    """
    Returns list of available locations extracted from columns.json.
    Used by frontend dropdown.
    """
    response = jsonify({"locations": util.get_location_names()})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# ------------------------------------------------------------
# API 2: Predict home price using ML model
# ------------------------------------------------------------
@app.route("/predict_home_price", methods=["GET", "POST"])
def predict_home_price():
    """
    Accepts form data from frontend:
        - total_sqft
        - location
        - bhk
        - bath
    Calls util.get_estimated_price() to compute predicted price.
    Returns JSON response.
    """
    total_sqft = float(request.form["total_sqft"])
    location = request.form["location"]
    bhk = int(request.form["bhk"])
    bath = int(request.form["bath"])

    estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)

    response = jsonify({"estimated_price": estimated_price})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")

    # Load saved model + columns + location data
    util.load_saved_artifacts()

    # Start Flask server
    app.run()
