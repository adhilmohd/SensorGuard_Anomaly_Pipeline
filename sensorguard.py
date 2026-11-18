import math
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_INDICES = {40, 120, 180, 220}
FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "temperature_change",
]


def generate_raw_data():
    timestamps = pd.date_range(
        start="2026-01-01",
        periods=240,
        freq="h",
    )
    records = []

    for index, timestamp in enumerate(timestamps):
        temperature = 50 + (0.02 * index) + (1.5 * math.sin(index / 8))
        vibration = 2 + (0.001 * index) + (0.15 * math.cos(index / 6))
        pressure = 30 + (0.3 * math.sin(index / 10))


        if index == 40:
            temperature += 20
        elif index == 120:
            vibration += 5
        elif index == 180:
            temperature += 15
            vibration += 4
        elif index == 220:
            temperature -= 20

        if index == 15:
            temperature = None
        if index == 75:
            vibration = -1

        records.append(
            {
                "timestamp": timestamp,
                "temperature_c": temperature,
                "vibration_mm_s": vibration,
                "pressure_bar": pressure,
                "ground_truth_anomaly": int(index in ANOMALY_INDICES),
            }
        )

    return records

def quality_report(records):
    missing_temperature = sum(
        row["temperature_c"] is None for row in records
    )
    invalid_vibration = sum(
        row["vibration_mm_s"] is not None
        and row["vibration_mm_s"] < 0
        for row in records
    )
    return {
        "missing_temperature": missing_temperature,
        "invalid_vibration": invalid_vibration,
    }

def clean_records(records):
    valid_temperatures = [
        row["temperature_c"]
        for row in records
        if row["temperature_c"] is not None
    ]
    valid_vibrations = [
        row["vibration_mm_s"]
        for row in records
        if row["vibration_mm_s"] is not None
        and row["vibration_mm_s"] >= 0
    ]
    temperature_mean = sum(valid_temperatures) / len(valid_temperatures)
    vibration_mean = sum(valid_vibrations) / len(valid_vibrations)

    cleaned = []
    for row in records:
        clean_row = row.copy()
        if clean_row["temperature_c"] is None:
            clean_row["temperature_c"] = temperature_mean
        if (
            clean_row["vibration_mm_s"] is None
            or clean_row["vibration_mm_s"] < 0
        ):
            clean_row["vibration_mm_s"] = vibration_mean
        cleaned.append(clean_row)

        for index, row in enumerate(cleaned):
          if index == 0:
            row["temperature_change"] = 0.0
          else:
            row["temperature_change"] = abs(
                row["temperature_c"]
                - cleaned[index - 1]["temperature_c"]
            )

    return cleaned

def plot_raw_data(raw_frame):
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(
        raw_frame["timestamp"],
        raw_frame["temperature_c"],
        label="Temperature",
    )
    axes[1].plot(
        raw_frame["timestamp"],
        raw_frame["vibration_mm_s"],
        color="orange",
        label="Vibration",
    )
    axes[0].set(title="Raw temperature", ylabel="Temperature (C)")
    axes[1].set(
        title="Raw vibration",
        xlabel="Timestamp",
        ylabel="Vibration (mm/s)",
    )
    axes[0].legend()
    axes[1].legend()
    figure.savefig("raw_telemetry.png", dpi=150, bbox_inches="tight")
    plt.show()
    
def persist_with_sql(clean_frame):
    connection = sqlite3.connect("sensorguard.db")
    clean_frame.to_sql(
        name="telemetry",
        con=connection,
        if_exists="replace",
        index=False,
    )
    summary = pd.read_sql(
        """
        SELECT ground_truth_anomaly, COUNT(*) AS row_count
        FROM telemetry
        GROUP BY ground_truth_anomaly
        ORDER BY ground_truth_anomaly
        """,
        connection,
    )
    connection.close()
    return summary

def naive_predictions(clean_frame):
    return [
        int(temperature > 60)
        for temperature in clean_frame["temperature_c"]
    ]

def compute_metrics(true_labels, predicted_labels):
    true_positives = sum(
        truth == 1 and prediction == 1
        for truth, prediction in zip(true_labels, predicted_labels)
    )
    false_positives = sum(
        truth == 0 and prediction == 1
        for truth, prediction in zip(true_labels, predicted_labels)
    )
    false_negatives = sum(
        truth == 1 and prediction == 0
        for truth, prediction in zip(true_labels, predicted_labels)
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

def build_model(clean_frame):
    model = IsolationForest(
        n_estimators=100,
        contamination=len(ANOMALY_INDICES) / len(clean_frame),
        random_state=42,
    )
    raw_predictions = model.fit_predict(clean_frame[FEATURE_COLUMNS])
    predictions = [
        int(prediction == -1) for prediction in raw_predictions
    ]
    anomaly_scores = [
        -score
        for score in model.decision_function(clean_frame[FEATURE_COLUMNS])
    ]
    return model, predictions, anomaly_scores

def plot_detection(clean_frame):
    detected = clean_frame[clean_frame["ml_anomaly"] == 1]
    figure, axis = plt.subplots(
        figsize=(12, 5),
        layout="constrained",
    )
    axis.plot(
        clean_frame["timestamp"],
        clean_frame["temperature_c"],
        label="Temperature",
    )
    axis.scatter(
        detected["timestamp"],
        detected["temperature_c"],
        color="red",
        s=50,
        label="ML anomaly",
    )
    axis.set(
        title="SensorGuard anomaly report",
        xlabel="Timestamp",
        ylabel="Temperature (C)",
    )
    axis.legend()
    figure.savefig("anomaly_report.png", dpi=150, bbox_inches="tight")
    plt.show()

def main():
    raw_records = generate_raw_data()
    print("Raw data quality:", quality_report(raw_records))

    raw_frame = pd.DataFrame(raw_records)
    raw_frame.to_csv("raw_telemetry.csv", index=False)
    plot_raw_data(raw_frame)

    cleaned_records = clean_records(raw_records)
    print("Cleaned data quality:", quality_report(cleaned_records))

    clean_frame = pd.DataFrame(cleaned_records)
    sql_summary = persist_with_sql(clean_frame)
    print("SQL row counts:")
    print(sql_summary)

    clean_frame["naive_anomaly"] = naive_predictions(clean_frame)
    true_labels = list(clean_frame["ground_truth_anomaly"])
    print(
        "Naive detector:",
        compute_metrics(true_labels, clean_frame["naive_anomaly"]),
    )

    model, predictions, anomaly_scores = build_model(clean_frame)
    clean_frame["ml_anomaly"] = predictions
    clean_frame["anomaly_score"] = anomaly_scores
    print(
        "IsolationForest detector:",
        compute_metrics(true_labels, clean_frame["ml_anomaly"]),
    )
    print("IsolationForest trees:", model.n_estimators)

    clean_frame.to_csv("clean_telemetry.csv", index=False)
    anomalies = clean_frame[clean_frame["ml_anomaly"] == 1]
    anomalies.to_csv("anomalies.csv", index=False)
    plot_detection(clean_frame)

    print(
        "Created raw_telemetry.csv, clean_telemetry.csv, "
        "anomalies.csv, sensorguard.db, raw_telemetry.png, "
        "and anomaly_report.png"
    )


if __name__ == "__main__":
    main()


