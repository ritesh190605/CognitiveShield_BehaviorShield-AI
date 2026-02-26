from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    scaler: StandardScaler
    log_reg: LogisticRegression
    random_forest: RandomForestClassifier
    isolation_forest: IsolationForest
    iso_threshold: float
    feature_columns: list
    metrics: Dict[str, Dict[str, Any]]


def _oversample_minority(X: np.ndarray, y: np.ndarray, seed: int) -> tuple:
    """Oversample minority class via random duplication (no external deps)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos >= n_neg or n_pos == 0:
        return X, y
    n_sample = n_neg - n_pos
    dup_idx = rng.choice(pos_idx, size=n_sample)
    X_res = np.vstack([X, X[dup_idx]])
    y_res = np.concatenate([y, y[dup_idx]])
    shuffle_idx = rng.permutation(len(y_res))
    return X_res[shuffle_idx], y_res[shuffle_idx]


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[ModelBundle, pd.DataFrame]:
    """
    Train Logistic Regression, Random Forest, and Isolation Forest.
    Handles class imbalance via oversampling on the training set.
    Returns:
        model_bundle
        evaluation_df (with y_true, preds and probabilities for analysis)
    """
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Oversample minority class on training data only
    X_arr = X_train.values
    y_arr = y_train.values
    X_train_res_arr, y_train_res = _oversample_minority(X_arr, y_arr, seed)
    X_train_res = pd.DataFrame(
        X_train_res_arr, columns=X_train.columns, index=range(len(X_train_res_arr))
    )

    # Scale numeric-style features
    scaler = StandardScaler(with_mean=False)
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    log_reg = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        random_state=seed,
    )
    log_reg.fit(X_train_res_scaled, y_train_res)
    lr_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    rf.fit(X_train_res, y_train_res)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    # Isolation Forest – train on non-fraud from training set
    X_train_nonfraud = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
    )
    iso.fit(X_train_nonfraud)

    # Decision function: lower scores are more anomalous
    iso_scores_test = iso.decision_function(X_test)
    # set anomaly threshold at 10th percentile of non-fraud scores
    nonfraud_scores = iso.decision_function(X_train_nonfraud)
    iso_threshold = float(np.percentile(nonfraud_scores, 10))
    iso_preds = (iso_scores_test < iso_threshold).astype(int)

    metrics = {
        "log_reg": _compute_metrics(y_test.values, lr_preds),
        "random_forest": _compute_metrics(y_test.values, rf_preds),
        "isolation_forest": _compute_metrics(y_test.values, iso_preds),
    }

    eval_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "lr_prob": lr_probs,
            "lr_pred": lr_preds,
            "rf_prob": rf_probs,
            "rf_pred": rf_preds,
            "iso_score": iso_scores_test,
            "iso_pred": iso_preds,
        }
    )

    bundle = ModelBundle(
        scaler=scaler,
        log_reg=log_reg,
        random_forest=rf,
        isolation_forest=iso,
        iso_threshold=iso_threshold,
        feature_columns=feature_columns,
        metrics=metrics,
    )

    return bundle, eval_df


def predict_for_features(
    bundle: ModelBundle, X_new: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict fraud probabilities and anomaly scores for new feature rows.
    """
    X_aligned = X_new[bundle.feature_columns]
    X_scaled = bundle.scaler.transform(X_aligned)

    lr_probs = bundle.log_reg.predict_proba(X_scaled)[:, 1]
    rf_probs = bundle.random_forest.predict_proba(X_aligned)[:, 1]
    iso_scores = bundle.isolation_forest.decision_function(X_aligned)
    iso_flags = (iso_scores < bundle.iso_threshold).astype(int)

    preds = pd.DataFrame(
        {
            "lr_prob": lr_probs,
            "rf_prob": rf_probs,
            "iso_score": iso_scores,
            "iso_flag": iso_flags,
        },
        index=X_new.index,
    )
    return preds


def save_bundle(bundle: ModelBundle, path: str = "fraud_model_bundle.joblib") -> None:
    dump(bundle, path)


def load_bundle(path: str = "fraud_model_bundle.joblib") -> ModelBundle:
    return load(path)


if __name__ == "__main__":
    from data_preprocessing import load_or_generate_dataset
    from feature_engineering import build_user_profiles, engineer_features

    df_raw = load_or_generate_dataset()
    profiles = build_user_profiles(df_raw)
    X, y = engineer_features(df_raw, profiles)
    bundle, eval_df = train_models(X, y)
    print("Metrics:", bundle.metrics)
    print(eval_df.head())

