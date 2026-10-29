from pathlib import Path
import io

import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.predict import (
    load_model,
    load_family_model,
    analyze_traffic,
    FEATURE_NAMES
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
CORS(app)

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DATA = PROJECT_DIR / "dataset" / "KDDTrain+.txt"
COLUMNS = FEATURE_NAMES + ["label", "difficulty"]
CATEGORICAL = {"protocol_type", "service", "flag"}

# Load both models once at application startup.
model = load_model()
family_model = load_family_model()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/single")
def single_page():
    return render_template("single.html")


@app.route("/batch")
def batch_page():
    return render_template("batch.html")


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "model": "Decision Tree",
        "attack_family_model": "Random Forest",
        "expected_features": len(FEATURE_NAMES)
    })


@app.route("/api/options", methods=["GET"])
def get_options():
    """Return real categorical values found in the training dataset."""
    try:
        data = pd.read_csv(
            TRAIN_DATA,
            header=None,
            names=COLUMNS,
            usecols=["protocol_type", "service", "flag"]
        )
        return jsonify({
            "protocol_type": sorted(data["protocol_type"].dropna().unique().tolist()),
            "service": sorted(data["service"].dropna().unique().tolist()),
            "flag": sorted(data["flag"].dropna().unique().tolist())
        })
    except (OSError, ValueError) as error:
        app.logger.exception("Could not load categorical options")
        return jsonify({"error": str(error)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request must contain a JSON object."}), 400

    features = data.get("features")
    if not isinstance(features, list):
        return jsonify({"error": "'features' must be an array."}), 400

    try:
        result = analyze_traffic(
            model,
            family_model,
            features,
            simplified=bool(data.get("simplified", False))
        )
        return jsonify(result)
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Single-traffic prediction failed")
        return jsonify({"error": "Prediction failed. Check the server logs."}), 500


@app.route("/api/predict-csv", methods=["POST"])
def predict_csv():
    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a CSV file."}), 400
    if not uploaded_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    try:
        content = uploaded_file.read()
        if not content:
            return jsonify({"error": "The uploaded file is empty."}), 400

        data = pd.read_csv(io.BytesIO(content), header=None, skipinitialspace=True)
        if data.empty:
            return jsonify({"error": "No records found in the CSV."}), 400

        # Detect a named header containing all feature names.
        first_row = [str(value).strip().lower() for value in data.iloc[0].tolist()]
        if set(FEATURE_NAMES).issubset(set(first_row)):
            data = pd.read_csv(io.BytesIO(content), skipinitialspace=True)
            data.columns = [str(column).strip().lower() for column in data.columns]
            if not set(FEATURE_NAMES).issubset(set(data.columns)):
                return jsonify({
                    "error": "The CSV is missing required feature columns."
                }), 400
            data = data[FEATURE_NAMES].copy()
        else:
            if data.shape[1] not in (41, 42, 43):
                return jsonify({
                    "error": (
                        "CSV must contain 41 features, or 42/43 columns "
                        "in NSL-KDD format."
                    )
                }), 400
            data = data.iloc[:, :41].copy()
            data.columns = FEATURE_NAMES

        if data.empty:
            return jsonify({"error": "No records found in the CSV."}), 400
        if len(data) > 5000:
            return jsonify({"error": "Upload at most 5,000 records per request."}), 400

        numeric_features = [name for name in FEATURE_NAMES if name not in CATEGORICAL]
        for name in numeric_features:
            data[name] = pd.to_numeric(data[name], errors="coerce")

        if data.isnull().any().any():
            return jsonify({
                "error": (
                    "The CSV contains missing or invalid values. "
                    "Check all 41 feature columns."
                )
            }), 400

        for name in CATEGORICAL:
            data[name] = data[name].astype(str).str.strip()
            if data[name].eq("").any():
                return jsonify({
                    "error": f"The {name} column contains empty values."
                }), 400

        results = []
        for index, (_, row) in enumerate(data.iterrows(), start=1):
            features = [row[name] for name in FEATURE_NAMES]
            result = analyze_traffic(model, family_model, features)
            results.append({
                "row": index,
                "prediction": result["prediction"],
                "label": result["label"],
                "attack_family": result["attack_family"],
                "explanation": result["explanation"],
                "recommendations": result["recommendations"],
                "family_note": result["family_note"]
            })

        attack_count = sum(item["prediction"] == 1 for item in results)
        return jsonify({
            "total_records": len(results),
            "normal_count": len(results) - attack_count,
            "attack_count": attack_count,
            "predictions": results,
            "note": (
                "Attack families are estimates, not confirmed attack types. "
                "R2L and U2R recall was low in the independent NSL-KDD test."
            )
        })

    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return jsonify({"error": f"Could not process the CSV: {error}"}), 400
    except Exception:
        app.logger.exception("CSV prediction failed")
        return jsonify({"error": "CSV prediction failed. Check the server logs."}), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
