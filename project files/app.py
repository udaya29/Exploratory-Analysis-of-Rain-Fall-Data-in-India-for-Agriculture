from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load saved files
model = pickle.load(open(r"C:\Users\DELL\OneDrive\Desktop\RFD\SM\rainfall.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\DELL\OneDrive\Desktop\RFD\SM\scaler.pkl", "rb"))
encoders = pickle.load(open(r"C:\Users\DELL\OneDrive\Desktop\RFD\SM\encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # 🔹 Convert numeric columns to float
        numeric_cols = [
            'MinTemp',
            'MaxTemp',
            'Rainfall',
            'Humidity9am',
            'Humidity3pm',
            'Pressure9am',
            'Pressure3pm'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 🔹 Encode categorical columns
        for col in encoders:
            if col in df.columns:
                try:
                    df[col] = encoders[col].transform(df[col])
                except Exception as e:
                     return f"Encoding Error for {col}: {e}"

        # 🔹 Scale
        scaled_data = scaler.transform(df)

        # 🔹 Predict
        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("noChance.html")

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
