import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Maximum 16MB file upload limit
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "nids_xgboost_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "nids_scaler.pkl")

os.makedirs('app/uploads', exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enterprise Network Intrusion Monitor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .container { max-width: 900px; margin: 30px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        .btn:hover { background: #2980b9; }
        .file-upload { margin: 20px 0; border: 2px dashed #bdc3c7; padding: 30px; text-align: center; border-radius: 8px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .attack { background-color: #ffcccc; color: #900; font-weight: bold; }
        .safe { color: #27ae60; }
        .error { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ ThreatWatch NIDS Scanner</h1>
        <p>Powered by XGBoost & Neural Trees</p>
    </div>
    
    <div class="container">
        <h2>Upload Network Telemetry Log (.CSV)</h2>
        <p>The system will scan each packet trace mathematically and flag anomalous, malicious behavior instantly.</p>
        
        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
        
        <form action="/" method="POST" enctype="multipart/form-data">
            <div class="file-upload">
                <input type="file" name="file" accept=".csv" required>
            </div>
            <button type="submit" class="btn">Scan Network Traffic</button>
        </form>

        {% if results is not none %}
        <h3>Scan Report</h3>
        <p>Scanned {{ results|length }} packets.</p>
        <table>
            <tr>
                <th>Traffic ID</th>
                <th>Status</th>
                <th>Confidence</th>
            </tr>
            {% for item in results %}
            <tr class="{% if item.is_attack %}attack{% else %}safe{% endif %}">
                <td>{{ loop.index }} #TRX-{{ item.id }}</td>
                <td>{% if item.is_attack %}Anomalous (Block){% else %}Legitimate (Pass){% endif %}</td>
                <td>{{ "%.2f"|format(item.confidence * 100) }}%</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
</body>
</html>
"""

def get_nids_models():
    """Lazily load NIDS Scaler & XGBoost Model."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None
    
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            path = os.path.join("app/uploads", secure_filename(file.filename))
            file.save(path)
            
            # Predict
            model, scaler = get_nids_models()
            if model is None:
                # Let's see EXACTLY why it failed in the browser
                error = f"ERROR Loading Model from disk. Path: {MODEL_PATH}."
            else:
                try:
                    df = pd.read_csv(path)
                    
                    # Assume users are uploading ONLY the numerical features expected by the scaler payload
                    try:
                        scaled_data = scaler.transform(df.values)
                        preds = model.predict(scaled_data)
                        probs = model.predict_proba(scaled_data)[:, 1]
                        
                        results = []
                        for i in range(min(50, len(preds))): # Limit UI visualization to 50 for performance
                            results.append({
                                'id': i,
                                'is_attack': bool(preds[i] == 1),
                                'confidence': probs[i] if preds[i] == 1 else 1-probs[i]
                            })
                    except Exception as ve:
                        error = f"Feature mismatch. Ensure uploaded CSV mirrors the 42 KDD Cup 99 numerical inputs. {ve}"
                except Exception as e:
                    error = f"Failed to parse CSV: {str(e)}"
                    
    return render_template_string(HTML_TEMPLATE, error=error, results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
