"""
Full-Stack FinTech Fraud Detection — Flask Backend
Serves SaaS frontend + fraud detection API.
"""

import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Import fraud detection modules
from data_preprocessing import load_or_generate_dataset
from feature_engineering import build_user_profiles, engineer_features
from models import ModelBundle, predict_for_features, train_models
from simulation import build_stream, next_batch
from explainability import explain_transaction

app = Flask(__name__, static_folder="saas-frontend", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

# Global state (loaded once at startup)
_df = None
_profiles = None
_X = None
_y = None
_bundle = None
_stream_df = None


def _load_models():
    global _df, _profiles, _X, _y, _bundle, _stream_df
    if _df is None:
        _df = load_or_generate_dataset()
        _profiles = build_user_profiles(_df)
        _X, _y = engineer_features(_df, _profiles)
        _bundle, _ = train_models(_X, _y)
        _stream_df = build_stream(_df, _X)


_load_models()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/overview")
def api_overview():
    """Dataset summary + model metrics."""
    fraud_rate = float(_df["is_fraud"].mean()) if "is_fraud" in _df.columns else 0.0
    recall = _bundle.metrics["random_forest"]["recall"]
    precision = _bundle.metrics["random_forest"]["precision"]
    f1 = _bundle.metrics["random_forest"]["f1"]
    accuracy = _bundle.metrics["random_forest"]["accuracy"]

    return jsonify({
        "total_transactions": len(_df),
        "fraud_rate": round(fraud_rate * 100, 2),
        "users": int(_df["user_id"].nunique()),
        "metrics": {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
        "model_performance": {
            "log_reg": _bundle.metrics["log_reg"],
            "random_forest": _bundle.metrics["random_forest"],
            "isolation_forest": _bundle.metrics["isolation_forest"],
        },
    })


@app.route("/api/simulate")
def api_simulate():
    """Get next batch of transactions with fraud predictions and explanations."""
    batch_size = int(request.args.get("batch_size", 5))
    idx = int(request.args.get("idx", 0))
    threshold = float(request.args.get("threshold", 0.6))

    batch_df, next_idx = next_batch(_stream_df, idx, batch_size=batch_size)

    feature_cols = _bundle.feature_columns
    feat_batch = batch_df[feature_cols]
    preds = predict_for_features(_bundle, feat_batch)

    transactions = []
    alerts = []

    for i in range(len(batch_df)):
        row = batch_df.iloc[i].to_dict()
        feat_row = feat_batch.iloc[i]
        pred_row = preds.iloc[i]
        profile = _profiles.get(row["user_id"])

        exp = explain_transaction(
            tx_row=batch_df.iloc[i],
            feat_row=feat_row,
            user_profile=profile,
            lr_prob=float(pred_row["lr_prob"]),
            rf_prob=float(pred_row["rf_prob"]),
            iso_score=float(pred_row["iso_score"]),
            iso_flag=int(pred_row["iso_flag"]),
            proba_threshold=threshold,
        )

        ts = _ts_str(row.get("timestamp"))
        tx = {
            "user_id": str(row.get("user_id", "")),
            "amount": float(row.get("amount", 0)),
            "timestamp": ts,
            "location": str(row.get("location", "")),
            "device_id": str(row.get("device_id", "")),
            "is_fraud": int(row.get("is_fraud", 0)),
            "fraud_prob": round(exp["fraud_probability"], 3),
            "iso_score": round(exp["iso_score"], 3),
            "flagged": exp["is_flagged"],
            "reasons": exp["reasons"],
        }
        transactions.append(tx)
        if exp["is_flagged"]:
            alerts.append(tx)

    return jsonify({
        "transactions": transactions,
        "alerts": alerts,
        "next_idx": next_idx,
    })


def _ts_str(ts):
    if ts is None:
        return ""
    return str(ts).split(".")[0][:19].replace("T", " ")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  FinTech Fraud Detection — Full-Stack")
    print("=" * 55)
    print("\n  Open: http://127.0.0.1:5000")
    print("\n  Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
