from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load trained model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Simple HTML UI
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Breast Cancer Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        input { margin: 5px; padding: 5px; width: 150px; }
        button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; margin-top: 10px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 8px; font-size: 1.2em; }
        .malignant { background: #ffcccc; color: #cc0000; }
        .benign { background: #ccffcc; color: #006600; }
        label { display: inline-block; width: 220px; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>🎗️ Breast Cancer Prediction</h1>
    <p>Enter the cell nucleus features to predict whether the tumor is <b>Malignant</b> or <b>Benign</b>.</p>
    <form id="predForm">
        {% for feature in features %}
        <div>
            <label>{{ feature }}:</label>
            <input type="number" step="any" name="{{ feature }}" required placeholder="0.0">
        </div>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('predForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = Object.fromEntries(new FormData(e.target));
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            });
            const data = await response.json();
            const div = document.getElementById('result');
            div.className = 'result ' + (data.prediction === 'Malignant' ? 'malignant' : 'benign');
            div.innerHTML = `<b>Prediction:</b> ${data.prediction} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`;
        });
    </script>
</body>
</html>
"""

# These are the features remaining after correlated feature removal (23 features)
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "texture_worst", "area_worst", "smoothness_worst",
    "concavity_worst", "symmetry_worst"
]


@app.route("/")
def index():
    return render_template_string(HTML, features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[float(data[f]) for f in FEATURES]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max()

        return jsonify({
            "prediction": "Malignant" if prediction == 1 else "Benign",
            "confidence": round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
