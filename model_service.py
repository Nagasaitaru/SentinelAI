from pathlib import Path
from datetime import datetime
import json
import math
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "ensemble_intrusion_model.pkl"
METRICS_PATH = BASE / "artifacts" / "metrics.json"
DATASET_PATH = BASE / "data" / "DrDoS_DNS_R2l_imbala.csv"

# IMPORTANT:
# The supplied PKL was trained with exactly these 14 features, in this order.
# The CSV contains one additional feature, flow_bytes_per_seconds, but that
# column is NOT part of the supplied PKL input schema.
FIXED_MODEL_FEATURES = [
    "protocol",
    "flow_duration",
    "total_forward_packets",
    "total_backward_packets",
    "total_forward_packets_length",
    "total_backward_packets_length",
    "forward_packet_length_mean",
    "backward_packet_length_mean",
    "forward_packets_per_second",
    "backward_packets_per_second",
    "forward_iat_mean",
    "backward_iat_mean",
    "flow_iat_mean",
    "flow_packets_per_seconds",
]


class IntrusionModelService:
    def __init__(self):
        self.loaded = False
        self.rf = None
        self.et = None
        self.lr = None
        self.scaler = None
        self.threshold = 0.5
        self.feature_names = list(FIXED_MODEL_FEATURES)
        self.history = []
        self.metrics = {}
        self.dataset = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else None

        self._validate_fixed_dataset()
        if MODEL_PATH.exists():
            self._load_model()
        else:
            raise FileNotFoundError(f"Supplied PKL not found: {MODEL_PATH}")
        self._load_metrics()

    def _validate_fixed_dataset(self):
        if self.dataset is None:
            raise FileNotFoundError(f"Bundled dataset not found: {DATASET_PATH}")

        missing = [c for c in self.feature_names if c not in self.dataset.columns]
        if missing:
            raise ValueError(
                "Fixed dataset is missing model feature(s): " + ", ".join(missing)
            )

        if "label" not in self.dataset.columns:
            raise ValueError("Fixed dataset must contain the 'label' column.")

    def _load_model(self):
        package = joblib.load(MODEL_PATH)
        self.rf = package["random_forest"]
        self.et = package["extra_trees"]
        self.lr = package["logistic_regression"]
        self.scaler = package["scaler"]
        self.threshold = float(package.get("threshold", 0.5))

        expected = len(self.feature_names)
        for model_name, model in (
            ("Random Forest", self.rf),
            ("Extra Trees", self.et),
            ("Logistic Regression", self.lr),
            ("Scaler", self.scaler),
        ):
            n_features = getattr(model, "n_features_in_", None)
            if n_features is not None and int(n_features) != expected:
                raise ValueError(
                    f"{model_name} expects {n_features} features, "
                    f"but the fixed PKL schema uses {expected}."
                )

        self.loaded = True

    def _load_metrics(self):
        if METRICS_PATH.exists():
            self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def _clean_values(self, values):
        if len(values) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, received {len(values)}."
            )

        cleaned = []
        for value in values:
            if value is None or value == "":
                raise ValueError("All feature values are required.")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Feature values must be finite numbers.")
            cleaned.append(number)

        # Keep the exact feature names expected by the supplied sklearn PKL.
        return pd.DataFrame([cleaned], columns=self.feature_names)

    def predict(self, values, record_history=True):
        X = self._clean_values(values)

        rf_prob = float(self.rf.predict_proba(X)[0, 1])
        et_prob = float(self.et.predict_proba(X)[0, 1])

        X_lr = self.scaler.transform(X)
        lr_prob = float(self.lr.predict_proba(X_lr)[0, 1])

        ensemble_prob = (rf_prob + et_prob + lr_prob) / 3.0
        prediction = int(ensemble_prob >= self.threshold)

        # Trust is a simple decision-support score derived from detection
        # output, matching the paper's implicit trust concept.
        trust = max(0.0, min(100.0, (1.0 - ensemble_prob) * 100.0))
        risk = (
            "HIGH"
            if ensemble_prob >= 0.70
            else ("MEDIUM" if ensemble_prob >= 0.40 else "LOW")
        )

        result = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "prediction": "Attack" if prediction else "Normal",
            "label": prediction,
            "attack_probability": round(ensemble_prob * 100, 2),
            "risk": risk,
            "trust_score": round(trust, 2),
            "status": "Detected" if prediction else "Cleared",
            "model": "Soft Voting",
            "base_models": {
                "Random Forest": round(rf_prob * 100, 2),
                "Extra Trees": round(et_prob * 100, 2),
                "Logistic Regression": round(lr_prob * 100, 2),
            },
        }

        if record_history:
            self.history.insert(0, result)
            self.history = self.history[:100]

        return result

    def overview(self):
        total = len(self.history)
        attacks = sum(x["label"] == 1 for x in self.history)
        normal = total - attacks

        return {
            "total_traffic": total,
            "normal_traffic": normal,
            "attacks_detected": attacks,
            "detection_accuracy": self.metrics.get("ensemble", {}).get("accuracy"),
            "f1": self.metrics.get("ensemble", {}).get("f1"),
            "recall": self.metrics.get("ensemble", {}).get("recall"),
            "roc_auc": self.metrics.get("ensemble", {}).get("roc_auc"),
            "current_risk": self.history[0]["risk"] if self.history else "LOW",
        }

    def model_metrics(self):
        return self.metrics

    def dataset_info(self):
        labels = self.dataset["label"].value_counts().to_dict()
        return {
            "filename": DATASET_PATH.name,
            "rows": int(len(self.dataset)),
            "features": len(self.feature_names),
            "csv_columns": len(self.dataset.columns) - 1,
            "feature_names": self.feature_names,
            "excluded_csv_feature": "flow_bytes_per_seconds",
            "labels": {str(k): int(v) for k, v in labels.items()},
            "binary_mapping": {"BENIGN": 0, "DrDoS_DNS": 1, "R2L": 1},
        }

    def _dataset_row_values(self, row_number):
        if row_number < 1 or row_number > len(self.dataset):
            raise ValueError(f"Row must be between 1 and {len(self.dataset)}.")

        row = self.dataset.iloc[row_number - 1]

        # Never construct Feature_1/Feature_2/etc. names.
        # Always use the exact fixed PKL schema above.
        values = pd.to_numeric(
            row.loc[self.feature_names], errors="coerce"
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Dataset row {row_number} contains a non-finite model input.")

        return row, values.tolist()

    def dataset_row(self, row_number):
        row, values = self._dataset_row_values(row_number)
        prediction = self.predict(values, record_history=True)
        return {
            "row": row_number,
            "original_label": str(row["label"]),
            "feature_names": self.feature_names,
            "features": values,
            "prediction": prediction,
        }

    def dataset_sample(self, count=10):
        count = min(count, len(self.dataset))
        results = []
        for idx in range(1, count + 1):
            row, values = self._dataset_row_values(idx)
            prediction = self.predict(values, record_history=False)
            results.append({
                "row": idx,
                "original_label": str(row["label"]),
                **prediction,
            })
        return results

    def clear_history(self):
        self.history.clear()
