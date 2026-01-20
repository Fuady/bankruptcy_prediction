import pickle
import pandas as pd
from flask import Flask, request, jsonify

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/bankruptcy_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


data_1 = {"x1": 0.20912,
 "x2": 0.49988,
 "x3": 0.47225,
 1.9447,
 14.786,
 0.0,
 0.25834,
 0.99601,
 1.6996,
 0.49788,
 0.26114,
 0.5168,
 0.15835,
 0.25834,
 677.96,
 0.53838,
 2.0005,
 0.25834,
 0.152,
 87.981,
 1.4293,
 0.24806,
 0.12304,
 0.540231131537013,
 0.39542,
 0.43992,
 88.444,
 16.946,
 3.6884,
 0.26969,
 0.152,
 122.17,
 2.9876,
 2.9876,
 0.20616,
 1.6996,
 173.45369449941683,
 0.49788,
 0.1213,
 0.086422,
 0.064371,
 0.14595,
 199.49,
 111.51,
 0.51045,
 1.1252,
 100.13,
 0.23727,
 0.13961,
 1.9447,
 0.49988,
 0.33472,
 17.866,
 17.866,
 2304.6,
 0.1213,
 0.42002,
 0.853,
 0.0,
 4.1486,
 3.2732,
 107.35,
 3.4,
 60.987}

# data_1 = {
#         "age": 41,
#         "incident_severity": "Total Loss",
#         "total_claim_amount": 71600,
#         "insured_hobbies": "chess",
#         "policy_state": "OH",
#         "number_of_vehicles_involved": 1,
#         "property_damage": "YES",
#         "auto_model": "92x",
#         "insured_occupation": "craft-repair",
#         "vehicle_claim": 52080,
#         "bodily_injuries": 1,
#         "months_as_customer": 328,
#         "insured_relationship": "husband",
#         "injury_claim": 6510,
#         "insured_zip": 466132,
#         "witnesses": 2,
#         "capital-loss": 0,
#         "authorities_contacted": "Police",
#         "property_claim": 13020,
#         "capital-gains": 53300,
#         "incident_type": "Single Vehicle Collision",
#         "insured_education_level": "MD",
#         "collision_type": "Side Collision",
#         "umbrella_limit": 0,
#         "policy_number": 521585,
#         "policy_csl": "250/500",
#         "insured_sex": "MALE",
#         "auto_year": 2004,
#         "auto_make": "Saab",
#         "policy_annual_premium": 1406.91,
#         "police_report_available": "YES",
#         "incident_hour_of_the_day": 5,
#         "policy_deductable": 1000
# }


# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Bankruptcy Prediction API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data_1 = request.get_json()

    X = pd.DataFrame([data_1])
    fraud_prob  = model.predict_proba(X)[0,1]
    fraud = fraud_prob >= 0.5

    results = {
            'Bankruptcy Probability': float(fraud_prob),
            'Bankruptcy': bool(fraud)
    }
    return jsonify(results)


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)