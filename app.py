from flask import Flask, render_template, request, jsonify
from model_service import IntrusionModelService
from pathlib import Path
import csv
import io
import pandas as pd

app = Flask(__name__)
service = IntrusionModelService()

@app.route("/")
def dashboard():
    return render_template("dashboard.html", active="dashboard")

@app.route("/live-detection")
def live_detection():
    return render_template("live_detection.html", active="live")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html", active="prediction")

@app.route("/model-performance")
def model_performance():
    return render_template("model_performance.html", active="performance")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", active="analytics")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html", active="dataset")

@app.route("/detection-history")
def detection_history():
    return render_template("history.html", active="history")

@app.route("/about")
def about():
    return render_template("about.html", active="about")

@app.route("/settings")
def settings():
    return render_template("settings.html", active="settings")

@app.get("/api/health")
def health():
    return jsonify({
        "status": "online" if service.loaded else "model_missing",
        "model_loaded": service.loaded,
        "model_type": "Random Forest + Extra Trees + Logistic Regression",
        "voting": "soft",
        "threshold": service.threshold
    })

@app.get("/api/overview")
def overview():
    return jsonify(service.overview())

@app.get("/api/detection-distribution")
def detection_distribution():
    # Full bundled dataset distribution: BENIGN is Normal; DrDoS_DNS and R2L are Attack.
    labels = service.dataset["label"].astype(str).str.strip().str.upper()
    normal = int((labels == "BENIGN").sum())
    attacks = int((labels != "BENIGN").sum())
    total = normal + attacks
    return jsonify({
        "normal": normal,
        "attacks": attacks,
        "total": total,
        "normal_percent": round(normal / total * 100, 2) if total else 0,
        "attack_percent": round(attacks / total * 100, 2) if total else 0
    })

@app.get("/api/dashboard-model-accuracies")
def dashboard_model_accuracies():
    # Classifier accuracies reported in the IEEE paper.
    # These are benchmark/report values and are kept separate from the
    # exact supplied PKL used for live dataset inference.
    return jsonify({
        "Logistic Regression": 0.988267,
        "SVM": 0.992000,
        "KNN": 0.999467,
        "Naive Bayes": 0.999733,
        "Random Forest": 0.998000,
        "Extra Trees": 0.964400
    })

@app.get("/api/model-metrics")
def model_metrics():
    return jsonify(service.model_metrics())

@app.get("/api/history")
def history():
    return jsonify(service.history)

@app.get("/api/features")
def features():
    return jsonify({"features": service.feature_names})

@app.get("/api/dataset-info")
def dataset_info():
    return jsonify(service.dataset_info())

@app.get("/api/dataset-row/<int:row_number>")
def dataset_row(row_number):
    try:
        return jsonify(service.dataset_row(row_number))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/api/dataset-sample")
def dataset_sample():
    try:
        return jsonify({"results": service.dataset_sample(10)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/history/clear")
def clear_history():
    service.clear_history()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
