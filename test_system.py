"""
test_system.py — Automated test suite for the fraud detection system.
Tests all modules: database, feature engineering, prediction, and simulator.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime


def test_database():
    """Test database operations."""
    print("=" * 60)
    print("TEST 1: Database Operations")
    print("=" * 60)

    from src.database_manager import DatabaseManager

    # Use temp database
    db = DatabaseManager(db_path="database/test_fraud.db")
    db.clear_all_data()

    # Test user profile creation
    profile = db.get_user_profile(9999)
    assert profile["user_id"] == 9999
    assert profile["avg_amount"] == 0.0
    print("  ✅ Get default user profile")

    # Test profile update
    db.update_user_profile(9999, 500.0, "Android_A", "Mumbai", datetime.now().isoformat())
    profile = db.get_user_profile(9999)
    assert profile["avg_amount"] == 500.0
    assert profile["last_device"] == "Android_A"
    assert profile["transaction_count"] == 1
    print("  ✅ Update user profile")

    # Test transaction insert
    txn = {
        "transaction_id": "TEST-001",
        "user_id": 9999,
        "amount": 500.0,
        "hour": 14,
        "device_id": "Android_A",
        "location": "Mumbai",
        "merchant_id": "paytm@upi",
        "fraud_probability": 0.15,
        "risk_level": "LOW RISK",
        "explanation": "Test transaction",
        "timestamp": datetime.now().isoformat(),
    }
    db.insert_transaction(txn)
    recent = db.get_recent_transactions(limit=10)
    assert len(recent) == 1
    assert recent[0]["transaction_id"] == "TEST-001"
    print("  ✅ Insert and retrieve transaction")

    # Test fraud stats
    stats = db.get_fraud_stats()
    assert stats["total_transactions"] == 1
    print("  ✅ Get fraud stats")

    # Test velocity
    velocity = db.get_transaction_velocity(9999, datetime.now().isoformat())
    assert velocity >= 1
    print("  ✅ Get transaction velocity")

    db.clear_all_data()
    print("  ✅ Clear database")
    print("  ✅ All database tests passed!\n")
    return True


def test_feature_engineering():
    """Test feature engineering."""
    print("=" * 60)
    print("TEST 2: Feature Engineering")
    print("=" * 60)

    from src.data_processing import (
        compute_behavioral_features,
        build_feature_dataframe,
        generate_explanation,
        FEATURE_COLUMNS,
    )

    transaction = {
        "user_id": 1001,
        "amount": 5000.0,
        "hour": 3,
        "device_id": "iPhone_X",
        "location": "Delhi",
        "merchant_id": "gpay@upi",
    }

    user_profile = {
        "avg_amount": 250.0,
        "last_device": "Android_A",
        "usual_location": "Mumbai",
        "transaction_count": 50,
    }

    features = compute_behavioral_features(transaction, user_profile, 3)
    print(f"  Computed {len(features)} features")

    # Check behavioral flags
    assert features["is_night"] == 1, "Should flag night (hour=3)"
    assert features["is_new_device"] == 1, "Should flag new device"
    assert features["location_change_flag"] == 1, "Should flag location change"
    assert features["amount_deviation"] > 1.0, "Should have high amount deviation"
    print("  ✅ Behavioral flags computed correctly")

    # Check one-hot encoding
    assert features["location_Delhi"] == 1
    assert features["location_Mumbai"] == 0
    assert features["device_id_iPhone_X"] == 1
    assert features["merchant_id_gpay@upi"] == 1
    print("  ✅ One-hot encoding correct")

    # Build DataFrame
    df = build_feature_dataframe(features)
    assert list(df.columns) == FEATURE_COLUMNS
    assert df.shape == (1, 24)
    print(f"  ✅ DataFrame shape: {df.shape} (matches model's 24 features)")

    # Test explanation
    explanation = generate_explanation(features, 0.85, 0.65)
    assert "HIGH RISK" in explanation
    assert "Unusual transaction amount" in explanation
    print("  ✅ Explanation generated correctly")
    print("  ✅ All feature engineering tests passed!\n")
    return True


def test_prediction():
    """Test prediction engine."""
    print("=" * 60)
    print("TEST 3: Prediction Engine")
    print("=" * 60)

    from src.fraud_prediction import predict_fraud, get_model_info

    # Model info
    info = get_model_info()
    print(f"  Model: {info['model_type']}")
    print(f"  Scaler: {info['scaler_type']}")
    print(f"  Threshold: {info['threshold']}")
    print(f"  Features: {info['n_features']}")
    print("  ✅ Model loaded successfully")

    # Normal transaction
    normal_txn = {
        "user_id": 1001,
        "amount": 250.0,
        "hour": 14,
        "device_id": "Android_A",
        "location": "Mumbai",
        "merchant_id": "paytm@upi",
    }
    normal_profile = {
        "avg_amount": 250.0,
        "last_device": "Android_A",
        "usual_location": "Mumbai",
        "transaction_count": 100,
    }

    result = predict_fraud(normal_txn, normal_profile, 1)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_level"] in ("HIGH RISK", "LOW RISK")
    print(f"  Normal txn → Prob: {result['fraud_probability']:.3f}, Risk: {result['risk_level']}")
    print("  ✅ Normal transaction predicted")

    # Suspicious transaction
    suspicious_txn = {
        "user_id": 1001,
        "amount": 15000.0,
        "hour": 2,
        "device_id": "iPhone_X",
        "location": "Kolkata",
        "merchant_id": "amazon@upi",
    }
    result2 = predict_fraud(suspicious_txn, normal_profile, 8)
    assert 0.0 <= result2["fraud_probability"] <= 1.0
    print(f"  Suspicious txn → Prob: {result2['fraud_probability']:.3f}, Risk: {result2['risk_level']}")
    print(f"  Explanation:\n{result2['explanation']}")
    print("  ✅ Suspicious transaction predicted")
    print("  ✅ All prediction tests passed!\n")
    return True


def test_simulator():
    """Test simulator."""
    print("=" * 60)
    print("TEST 4: Transaction Simulator")
    print("=" * 60)

    from src.database_manager import DatabaseManager
    from src.simulator import run_simulator

    db = DatabaseManager(db_path="database/test_fraud.db")
    db.clear_all_data()

    results = []

    def callback(result, count):
        results.append(result)
        risk = "🚨" if result["risk_level"] == "HIGH RISK" else "✅"
        print(f"  {risk} Txn #{count}: User {result['user_id']}, "
              f"₹{result['amount']:.2f}, {result['risk_level']} "
              f"(prob: {result['fraud_probability']:.2%})")

    count = run_simulator(db, num_transactions=10, delay=0.1, fraud_ratio=0.3, callback=callback)
    assert count == 10
    print(f"\n  Generated {count} transactions")

    # Verify in database
    recent = db.get_recent_transactions(limit=20)
    assert len(recent) == 10
    print(f"  ✅ {len(recent)} transactions stored in database")

    # Check stats
    stats = db.get_fraud_stats()
    print(f"  Total: {stats['total_transactions']}, "
          f"High Risk: {stats['high_risk_count']}, "
          f"Fraud Rate: {stats['fraud_rate']:.1f}%")
    print("  ✅ All simulator tests passed!\n")

    db.clear_all_data()
    return True


if __name__ == "__main__":
    print("\n🛡️  FINTECHAI — Automated System Tests\n")

    all_passed = True
    for test in [test_database, test_feature_engineering, test_prediction, test_simulator]:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready.")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)
