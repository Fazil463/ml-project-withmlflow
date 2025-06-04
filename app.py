from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline  # adjust import if needed
import os

app = Flask(__name__)

# Instantiate prediction pipeline once
prediction_pipeline = PredictionPipeline()

@app.route('/', methods=['GET'])
def home_page():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def train_model():
    os.system("python main.py")  # Your training script
    return "✅ Training Completed Successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs from request
        data = {
            'Gender': request.form['Gender'],
            'AGE': float(request.form['AGE']),
            'Urea': float(request.form['Urea']),
            'Cr': float(request.form['Cr']),
            'HbA1c': float(request.form['HbA1c']),
            'Chol': float(request.form['Chol']),
            'TG': float(request.form['TG']),
            'HDL': float(request.form['HDL']),
            'LDL': float(request.form['LDL']),
            'VLDL': float(request.form['VLDL']),
            'BMI': float(request.form['BMI']),
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Add missing dummy features
        input_df['ID'] = 0
        input_df['No_Pation'] = 0

        # Encode Gender (M=1, F=0)
        input_df['Gender'] = input_df['Gender'].map({'M': 1, 'F': 0})

        # Arrange columns in correct order expected by model
        input_df = input_df[['ID', 'No_Pation', 'Gender', 'AGE', 'Urea', 'Cr', 'HbA1c',
                             'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]

        # Get raw prediction from model
        prediction_raw = prediction_pipeline.predict(input_df)[0]
        print("DEBUG: Raw model prediction output:", prediction_raw)

        # If output is list/array (like probabilities), pick highest
        if isinstance(prediction_raw, (list, np.ndarray)):
            pred_class = np.argmax(prediction_raw)
        else:
            # Convert to integer label (round if float)
            pred_class = int(round(prediction_raw))

        # Map prediction class to label
        mapping = {0: 'Non-Diabetic', 1: 'Prediabetic', 2: 'Diabetic'}
        prediction_label = mapping.get(pred_class, "Unknown")

        # Render the results.html with prediction label
        return render_template('results.html', prediction=prediction_label)

    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, )
