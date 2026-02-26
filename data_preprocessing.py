import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATA_PATH_DEFAULT = Path("transactions_synthetic.csv")


def load_raw_data(csv_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load an existing transactions CSV if available.
    Expected minimal columns:
        user_id, amount, timestamp, device_id, location, is_fraud
    """
    path = Path(csv_path) if csv_path else DATA_PATH_DEFAULT
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def generate_synthetic_data(
    n_users: int = 200,
    n_transactions: int = 10000,
    seed: int = 42,
    fraud_ratio: float = 0.03,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset with behavioral structure.

    - Each user has typical amount range, active hours, home location, and 1–3 devices.
    - Fraudulent transactions tend to:
        * Have much higher amounts
        * Occur at odd hours
        * Come from new locations/devices
        * Appear in bursts
    """
    rng = np.random.default_rng(seed)

    user_ids = [f"U{uid:04d}" for uid in range(n_users)]

    # User-level behavior priors
    user_base_amount = rng.uniform(300, 3000, size=n_users)
    user_amount_std = rng.uniform(100, 800, size=n_users)
    user_active_start = rng.integers(7, 12, size=n_users)  # typical start hour
    user_active_end = rng.integers(18, 23, size=n_users)  # typical end hour

    cities = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata"]
    user_home_city = rng.choice(cities, size=n_users)
    user_device_counts = rng.integers(1, 4, size=n_users)

    records = []

    start_date = dt.datetime.now() - dt.timedelta(days=30)

    for _ in range(n_transactions):
        uid_idx = rng.integers(0, n_users)
        uid = user_ids[uid_idx]

        # Draw timestamp around the last 30 days
        days_offset = rng.uniform(0, 30)
        base_time = start_date + dt.timedelta(days=float(days_offset))

        # Decide if this transaction is fraudulent
        is_fraud = rng.random() < fraud_ratio

        # Non-fraud: stick to behavior; Fraud: deviate on multiple axes with some probability
        if not is_fraud:
            hour = int(
                rng.integers(user_active_start[uid_idx], user_active_end[uid_idx] + 1)
            )
            minute = int(rng.integers(0, 60))
            timestamp = base_time.replace(hour=hour, minute=minute, second=0)

            amount = float(
                rng.normal(user_base_amount[uid_idx], user_amount_std[uid_idx])
            )
            amount = max(50, amount)

            # Mostly home city
            if rng.random() < 0.9:
                location = user_home_city[uid_idx]
            else:
                location = rng.choice(cities)

            device_count = user_device_counts[uid_idx]
            device_id = f"D{uid}_{rng.integers(1, device_count + 1)}"
        else:
            # Fraud behavior: odd hours, high amount, new city/device with higher probability
            if rng.random() < 0.7:
                # Odd-hour fraud
                hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
            else:
                hour = int(
                    rng.integers(user_active_start[uid_idx], user_active_end[uid_idx] + 1)
                )
            minute = int(rng.integers(0, 60))
            timestamp = base_time.replace(hour=hour, minute=minute, second=0)

            # Larger amount spikes
            multiplier = rng.uniform(3, 10)
            amount = float(
                max(
                    100,
                    user_base_amount[uid_idx] + multiplier * user_amount_std[uid_idx],
                )
            )

            # More likely to be new or rare city
            if rng.random() < 0.6:
                # new / unusual city
                other_cities = [c for c in cities if c != user_home_city[uid_idx]]
                location = rng.choice(other_cities)
            else:
                location = user_home_city[uid_idx]

            device_count = user_device_counts[uid_idx]
            if rng.random() < 0.6:
                # new device id
                new_suffix = device_count + rng.integers(1, 4)
                device_id = f"D{uid}_{new_suffix}"
            else:
                device_id = f"D{uid}_{rng.integers(1, device_count + 1)}"

        records.append(
            {
                "user_id": uid,
                "amount": round(amount, 2),
                "timestamp": timestamp.isoformat(),
                "device_id": device_id,
                "location": location,
                "is_fraud": int(is_fraud),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Parse timestamps
    - Drop rows with critical missing values
    - Ensure correct dtypes
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Drop rows missing core fields
    core_cols = ["user_id", "amount", "timestamp", "device_id", "location"]
    df = df.dropna(subset=[c for c in core_cols if c in df.columns])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].astype(int)
    else:
        df["is_fraud"] = 0

    return df


def load_or_generate_dataset(
    csv_path: Optional[str] = None,
    n_users: int = 200,
    n_transactions: int = 10000,
) -> pd.DataFrame:
    """
    Main entrypoint for the rest of the app.
    - Try to load existing CSV.
    - If missing, generate synthetic data and return it.
    """
    df = load_raw_data(csv_path)
    if df is None:
        df = generate_synthetic_data(n_users=n_users, n_transactions=n_transactions)
    df = clean_and_prepare(df)
    return df


if __name__ == "__main__":
    data = load_or_generate_dataset()
    print(data.head())
    print(data["is_fraud"].value_counts(normalize=True))

