from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class UserProfile:
    user_id: str
    avg_amount: float
    std_amount: float
    typical_start_hour: float
    typical_end_hour: float
    home_locations: List[str]
    known_devices: List[str]
    avg_tx_per_hour: float


def build_user_profiles(df: pd.DataFrame) -> Dict[str, UserProfile]:
    """
    Build per-user behavioral profiles from historical data.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    profiles: Dict[str, UserProfile] = {}

    for user_id, g in df.groupby("user_id"):
        avg_amount = float(g["amount"].mean())
        std_amount = float(g["amount"].std(ddof=1) or 1.0)

        # Time window of typical activity – use 10th–90th percentile hours
        start_hour = float(np.nanpercentile(g["hour"], 10))
        end_hour = float(np.nanpercentile(g["hour"], 90))

        # Common locations (top 2)
        home_locations = (
            g["location"]
            .value_counts()
            .head(2)
            .index.astype(str)
            .tolist()
        )

        # Known devices
        known_devices = g["device_id"].dropna().astype(str).unique().tolist()

        # Approximate transactions per hour across the dataset
        g_sorted = g.sort_values("timestamp")
        if len(g_sorted) > 1:
            total_hours = (
                (g_sorted["timestamp"].iloc[-1] - g_sorted["timestamp"].iloc[0])
                .total_seconds()
                / 3600.0
            )
            total_hours = max(total_hours, 1e-3)
            avg_tx_per_hour = float(len(g_sorted) / total_hours)
        else:
            avg_tx_per_hour = 1.0

        profiles[user_id] = UserProfile(
            user_id=user_id,
            avg_amount=avg_amount,
            std_amount=std_amount,
            typical_start_hour=start_hour,
            typical_end_hour=end_hour,
            home_locations=home_locations,
            known_devices=known_devices,
            avg_tx_per_hour=avg_tx_per_hour,
        )

    return profiles


def _compute_velocity(
    df: pd.DataFrame, window_minutes: int = 30
) -> pd.Series:
    """
    Compute transaction count in a rolling time window per user.
    """
    df = df.sort_values("timestamp")
    window = f"{window_minutes}min"
    # groupby user and rolling count
    counts = (
        df.set_index("timestamp")
        .groupby("user_id")["amount"]
        .rolling(window=window)
        .count()
        .reset_index(level=0, drop=True)
    )
    return counts.reindex(df.index)


def engineer_features(
    df: pd.DataFrame,
    user_profiles: Dict[str, UserProfile],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create behavioral features at transaction-level.
    Returns:
        X: feature matrix
        y: labels (if available, otherwise zeros)
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    # Velocity feature: number of tx in last 30 minutes
    df["tx_count_30min"] = _compute_velocity(df, window_minutes=30).fillna(1.0)

    # Initialize behavioral deviation features
    df["amount_zscore"] = 0.0
    df["is_new_location"] = 0
    df["is_new_device"] = 0
    df["is_unusual_hour"] = 0
    df["velocity_ratio"] = 1.0

    for idx, row in df.iterrows():
        user_id = row["user_id"]
        profile = user_profiles.get(user_id)
        if not profile:
            continue

        # Amount z-score
        z = (row["amount"] - profile.avg_amount) / (profile.std_amount or 1.0)
        df.at[idx, "amount_zscore"] = float(z)

        # New location
        loc = str(row["location"])
        df.at[idx, "is_new_location"] = int(loc not in profile.home_locations)

        # New device
        dev = str(row["device_id"])
        df.at[idx, "is_new_device"] = int(dev not in profile.known_devices)

        # Unusual hour (outside 1h padding around typical window)
        h = row["hour"]
        pad = 1.0
        if (h < profile.typical_start_hour - pad) or (
            h > profile.typical_end_hour + pad
        ):
            df.at[idx, "is_unusual_hour"] = 1

        # Velocity ratio vs typical tx/hour
        tx_count = df.at[idx, "tx_count_30min"]
        expected_in_window = max(profile.avg_tx_per_hour * (30.0 / 60.0), 0.1)
        df.at[idx, "velocity_ratio"] = float(tx_count / expected_in_window)

    # Encode categorical variables in a simple way (one-hot)
    cat_cols = ["location"]
    df_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols, drop_first=True)

    feature_cols = [
        "amount",
        "hour",
        "tx_count_30min",
        "amount_zscore",
        "is_new_location",
        "is_new_device",
        "is_unusual_hour",
        "velocity_ratio",
    ]

    X_num = df[feature_cols]
    X = pd.concat([X_num, df_cat], axis=1)

    if "is_fraud" in df.columns:
        y = df["is_fraud"].astype(int)
    else:
        y = pd.Series(0, index=df.index, name="is_fraud")

    return X, y


if __name__ == "__main__":
    from data_preprocessing import load_or_generate_dataset

    data = load_or_generate_dataset()
    profiles = build_user_profiles(data)
    X, y = engineer_features(data, profiles)
    print(X.head())
    print("Features shape:", X.shape, "Fraud rate:", y.mean())

