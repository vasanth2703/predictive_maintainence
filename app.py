from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime, timezone
import requests
import json
import csv
import warnings
import os
from collections.abc import Sequence
from collections import OrderedDict

app = Flask(__name__)

# Initialize CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://192.168.128.128:3000"]}})  # Allow frontend origin

# Load the pipeline and feature list
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pipeline = joblib.load(os.path.join(BASE_DIR, "alert_prediction_pipeline.joblib"))

    feature_list = joblib.load(os.path.join(BASE_DIR, "feature_list.joblib"))
    print("Model and feature list loaded successfully.")
    print(f"Loading pipeline from: {os.path.join(BASE_DIR, 'event_prediction_pipeline.joblib')}")
    print(f"Loading feature list from: {os.path.join(BASE_DIR, 'feature_list.joblib')}")
except Exception as e:
    warnings.warn(f"Error loading model or feature list: {e}")
    pipeline = None
    feature_list = None

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

@app.route('/predict_ioda', methods=['GET'])
def predict_ioda():
    print("Received request for prediction")  # Debug print statement
    if pipeline is None or feature_list is None:
        return jsonify({'error': "Model or feature list failed to load."})

    try:
        # IODA API Parameters
        country_code = request.args.get('country_code', 'CG')
        start_time = request.args.get('start_time', '2024-12-27 21:00:00')
        end_time = request.args.get('end_time', '2024-12-28 21:00:00')

        # Convert times to Unix timestamps
        from_time = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        until_time = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())

        # IODA API URL
        api_url = f"https://api.ioda.inetintel.cc.gatech.edu/v2/outages/alerts?entityType=country&entityCode={country_code}&from={from_time}&until={until_time}"

        # Make the API Request
        response = requests.get(api_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        ioda_data = response.json().get('data', [])  # Extract the 'data' part
        print(f"IODA Data: {ioda_data}")  # Debug print statement

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(ioda_data)

        # Preprocess data
        df = preprocess_ioda_data(df, feature_list)

        # Handle the case where df is empty or has no rows
        if df.empty:
            return jsonify({'error': "No valid data to process in the IODA API response."})

        # Make Prediction (assuming you want to predict on the *first* row)
        prediction = pipeline.predict_proba(df)[0, 1]

        print(f"Prediction value: {prediction}")  # Debug print statement
        risk_level = "Low" if prediction < 0.4 else "Medium" if prediction < 0.7 else "High"
        message = f"The predicted risk of an outage is {'Low' if prediction < 0.4 else 'Medium' if prediction < 0.7 else 'High'}. Take necessary precautions."
        return jsonify({
            'prediction_probability': float(prediction),
            'risk_level': risk_level,
            'message': message
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error accessing IODA API: {e}"})
    except Exception as e:
        return jsonify({'error': f"Error processing data: {e}"})

def preprocess_ioda_data(df, feature_list):
    # Rename time column to start
    df = df.rename(columns={'time': 'start'})

    # Convert 'start' to datetime and other operations
    df['start'] = pd.to_datetime(df['start'], unit='s')  # Assuming 'time' is in seconds

    # Datetime Features (same as training)
    df['hour'] = df['start'].dt.hour
    df['dayofweek'] = df['start'].dt.dayofweek

    # Value Difference
    df['value_diff'] = df['historyValue'].fillna(0) - df['value'].fillna(0)

    # One-Hot Encode Categorical Features (Same as training)
    df = pd.get_dummies(df, columns=['datasource', 'level', 'condition'], dummy_na=True)

    # Reindex columns to match the training data
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0  # Add missing columns
    df = df[feature_list]  # Ensure correct order and selection

    return df

if __name__ == '__main__':
    app.run(debug=True)
