import os
import sys
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from waitress import serve

# Add base directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from src.data_processing import ChurnDataPreprocessor

app = Flask(__name__)

# Paths to trained model and preprocessor artifacts
artifacts_dir = os.path.join(base_dir, 'artifacts')
model_path = os.path.join(artifacts_dir, 'best_model.pkl')
preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')

model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        model = joblib.load(model_path)
        preprocessor = ChurnDataPreprocessor.load(preprocessor_path)
        print("Model and preprocessor successfully loaded into Flask application.")
    else:
        print("Warning: Model or preprocessor artifacts not found. Please train model first.")

@app.before_request
def initialize():
    if model is None or preprocessor is None:
        load_artifacts()

@app.route('/health', methods=['GET'])
def health():
    status = "healthy" if (model is not None and preprocessor is not None) else "uninitialized"
    return jsonify({
        "status": status,
        "service": "AI-Based Telecom Customer Churn Prediction API",
        "version": "1.0.0"
    })


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        load_artifacts()
        if model is None or preprocessor is None:
            return jsonify({"error": "Model artifacts not loaded. Train the model first."}), 500
            
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        return jsonify({"error": "Provide one customer object or a non-empty list."}), 400
    if any(not isinstance(record, dict) for record in data):
        return jsonify({"error": "Every prediction input must be a customer object."}), 400

    required_columns = {
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges',
    }
    missing_by_record = [
        sorted(required_columns - record.keys())
        for record in data
    ]
    if any(missing_by_record):
        return jsonify({
            "error": "Each customer must include all required fields.",
            "missing_fields": sorted(set().union(*map(set, missing_by_record))),
        }), 400

    input_df = pd.DataFrame(data)
    try:
        X_transformed = preprocessor.transform(input_df)
        predictions = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed)[:, 1]
    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid customer input: {error}"}), 400

    results = []
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        risk_level = (
            "High Risk" if proba > 0.65
            else ("Medium Risk" if proba > 0.40 else "Low Risk")
        )
        retention_offer = (
            "Offer 20% Discount + Free Tech Support" if risk_level == "High Risk"
            else (
                "Offer 1-Year Contract Upgrade with 10% Discount"
                if risk_level == "Medium Risk"
                else "Standard Customer Maintenance"
            )
        )

        cust_id = input_df.iloc[i].get('customerID', f'CUST-REQ-{i+1}')
        results.append({
            "customerID": cust_id,
            "churn_prediction": "Yes" if pred == 1 else "No",
            "churn_probability": round(float(proba), 4),
            "risk_level": risk_level,
            "recommended_action": retention_offer,
        })

    return jsonify({
        "status": "success",
        "predictions_count": len(results),
        "predictions": results,
    })

if __name__ == '__main__':
    load_artifacts()
    port = int(os.environ.get('PORT', 5000))
    serve(app, host='0.0.0.0', port=port)
