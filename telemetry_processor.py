"""
IoT Smart Factory Telemetry & Predictive Maintenance Pipeline
Author: Prashant Kumar
Description: Analyzes real-time smart factory machine sensors (temperature, vibration, pressure)
to detect operational anomalies and predict equipment failure before catastrophic breakdown.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def simulate_iot_telemetry_stream(n_samples=5000):
    np.random.seed(42)
    timestamps = pd.date_range("2026-01-01", periods=n_samples, freq="1min")
    
    # Base operational ranges
    vibration_hz = np.random.normal(45, 8, n_samples)
    temp_c = np.random.normal(72, 12, n_samples)
    pressure_psi = np.random.normal(120, 15, n_samples)
    rpm = np.random.normal(1800, 150, n_samples)
    
    # Introduce mechanical wear anomalies
    anomaly_mask = np.random.rand(n_samples) < 0.08
    vibration_hz[anomaly_mask] += np.random.uniform(25, 40, size=anomaly_mask.sum())
    temp_c[anomaly_mask] += np.random.uniform(20, 35, size=anomaly_mask.sum())
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'vibration_hz': vibration_hz,
        'temperature_c': temp_c,
        'pressure_psi': pressure_psi,
        'motor_rpm': rpm,
        'machine_failure': anomaly_mask.astype(int)
    })
    
    # Rolling feature engineering
    df['vib_rolling_mean'] = df['vibration_hz'].rolling(window=10, min_periods=1).mean()
    df['temp_rolling_std'] = df['temperature_c'].rolling(window=10, min_periods=1).std().fillna(0)
    
    return df

def run_iot_anomaly_engine():
    print("[+] Simulating Industrial IoT Telemetry Sensor Stream...")
    df = simulate_iot_telemetry_stream(8000)
    
    print(f"[+] Total Telemetry Records: {len(df):,}")
    print(f"[+] Failure Events Detected: {df['machine_failure'].sum()} ({df['machine_failure'].mean()*100:.2f}%)")
    
    X = df[['vibration_hz', 'temperature_c', 'pressure_psi', 'motor_rpm', 'vib_rolling_mean', 'temp_rolling_std']]
    y = df['machine_failure']
    
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    
    print("=" * 55)
    print(f"Predictive Maintenance ROC-AUC: {roc_auc_score(y_test, probas):.4f}")
    print("=" * 55)
    print("
Fault Detection Report:")
    print(classification_report(y_test, preds, target_names=["Normal Ops", "Failure Imminent"]))

if __name__ == '__main__':
    run_iot_anomaly_engine()
