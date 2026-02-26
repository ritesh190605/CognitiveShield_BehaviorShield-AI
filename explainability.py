from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from feature_engineering import UserProfile


def _format_hour_range(start: float, end: float) -> str:
    def _h(h: float) -> str:
        h_int = int(round(h)) % 24
        return f"{h_int:02d}:00"

    return f"{_h(start)}–{_h(end)}"


def explain_transaction(
    tx_row: pd.Series,
    feat_row: pd.Series,
    user_profile: Optional[UserProfile],
    lr_prob: float,
    rf_prob: float,
    iso_score: float,
    iso_flag: int,
    proba_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Generate a human-readable explanation for a single transaction.
    """
    reasons: List[str] = []

    # Ensemble-style main score: average of LR and RF
    main_prob = float((lr_prob + rf_prob) / 2.0)
    flagged = bool((main_prob >= proba_threshold) or bool(iso_flag))

    # Amount deviation
    amount_z = float(feat_row.get("amount_zscore", 0.0))
    if user_profile is not None:
        if amount_z > 3:
            reasons.append(
                f"Amount is {amount_z:.1f}σ higher than this user's typical spend "
                f"(avg ~₹{user_profile.avg_amount:.0f})."
            )
        elif amount_z < -2:
            reasons.append(
                "Amount is unusually low compared to this user's norm."
            )

    # New location
    if int(feat_row.get("is_new_location", 0)) == 1 and user_profile is not None:
        home_str = ", ".join(user_profile.home_locations) or "usual locations"
        reasons.append(
            f"New location detected: {tx_row.get('location')} "
            f"(user usually transacts in {home_str})."
        )

    # New device
    if int(feat_row.get("is_new_device", 0)) == 1 and user_profile is not None:
        reasons.append(
            f"New device ID `{tx_row.get('device_id')}` not seen before for this user."
        )

    # Unusual hour
    if int(feat_row.get("is_unusual_hour", 0)) == 1 and user_profile is not None:
        hr = int(tx_row["timestamp"].hour)
        reasons.append(
            f"Transaction at {hr:02d}:00, outside the user's usual active window "
            f"({_format_hour_range(user_profile.typical_start_hour, user_profile.typical_end_hour)})."
        )

    # Velocity / frequency spike
    vel_ratio = float(feat_row.get("velocity_ratio", 1.0))
    if vel_ratio > 2.0:
        reasons.append(
            f"High transaction velocity: ~{vel_ratio:.1f}× the user's normal rate "
            "in the recent time window."
        )

    # Isolation Forest anomaly
    if iso_flag:
        reasons.append(
            "Overall behavior of this transaction is anomalous compared to the user's normal pattern "
            f"(Isolation Forest score {iso_score:.3f})."
        )

    if not reasons:
        if flagged:
            reasons.append(
                "Model ensemble considers this transaction risky based on subtle behavioral deviations."
            )
        else:
            reasons.append("No strong anomaly indicators compared to the user's history.")

    explanation = {
        "fraud_probability": main_prob,
        "fraud_probability_lr": float(lr_prob),
        "fraud_probability_rf": float(rf_prob),
        "iso_score": float(iso_score),
        "iso_flag": int(iso_flag),
        "is_flagged": flagged,
        "reasons": reasons,
    }
    return explanation


def batch_explanations(
    df_tx: pd.DataFrame,
    df_feat: pd.DataFrame,
    user_profiles: Dict[str, UserProfile],
    preds_df: pd.DataFrame,
    proba_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Convenience helper to generate explanations for aligned dataframes.
    """
    explanations: List[Dict[str, Any]] = []
    for idx in df_tx.index:
        tx_row = df_tx.loc[idx]
        feat_row = df_feat.loc[idx]
        profile = user_profiles.get(tx_row["user_id"])
        row_pred = preds_df.loc[idx]

        exp = explain_transaction(
            tx_row=tx_row,
            feat_row=feat_row,
            user_profile=profile,
            lr_prob=float(row_pred["lr_prob"]),
            rf_prob=float(row_pred["rf_prob"]),
            iso_score=float(row_pred["iso_score"]),
            iso_flag=int(row_pred["iso_flag"]),
            proba_threshold=proba_threshold,
        )
        explanations.append(exp)
    return explanations


if __name__ == "__main__":
    from data_preprocessing import load_or_generate_dataset
    from feature_engineering import build_user_profiles, engineer_features
    from models import predict_for_features, train_models

    df = load_or_generate_dataset()
    profiles = build_user_profiles(df)
    X, y = engineer_features(df, profiles)
    bundle, _ = train_models(X, y)

    preds = predict_for_features(bundle, X.head(10))
    expls = batch_explanations(
        df_tx=df.head(10),
        df_feat=X.head(10),
        user_profiles=profiles,
        preds_df=preds,
    )
    for i, e in enumerate(expls, start=1):
        print(f"TX {i}: flagged={e['is_flagged']}, prob={e['fraud_probability']:.2f}")
        for r in e["reasons"]:
            print(" -", r)
        print()

