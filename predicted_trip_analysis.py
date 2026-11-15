import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib


BEHAVIOR_CATEGORIES = ["NORMAL", "AGGRESSIVE", "DROWSY"]
EVENT_NAMES = {
    0: "Normal",
    1: "Braking",
    2: "Turning",
    3: "Acceleration",
}
EVENT_RATE_COLUMNS = {
    "Braking": ("actual_braking_rate_pct", "predicted_braking_rate_pct"),
    "Acceleration": (
        "actual_acceleration_rate_pct",
        "predicted_acceleration_rate_pct",
    ),
    "Turning": ("actual_turning_rate_pct", "predicted_turning_rate_pct"),
    "Non-normal": (
        "actual_non_normal_event_rate_pct",
        "predicted_non_normal_event_rate_pct",
    ),
}
FORBIDDEN_FEATURES = {
    "behavior",
    "trip",
    "driver",
    "road",
    "label",
    "label_name",
}
OUTPUT_FILES = {
    "trip_rates": "predicted_trip_event_rates_heldout.csv",
    "behavior_summary": "predicted_trip_behavior_rate_summary.csv",
    "error_summary": "predicted_trip_rate_error_summary.csv",
    "analysis": "predicted_trip_analysis_summary.json",
    "behavior_plot": "predicted_vs_actual_non_normal_rate_by_behavior.png",
    "trip_plot": "predicted_vs_actual_non_normal_rate_by_trip.png",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_event_rates(labels):
    total = len(labels)
    counts = labels.value_counts()
    braking = int(counts.get(1, 0))
    turning = int(counts.get(2, 0))
    acceleration = int(counts.get(3, 0))

    def rate(count):
        return count / total * 100 if total else float("nan")

    return {
        "total_windows": total,
        "normal_windows": int(counts.get(0, 0)),
        "braking_windows": braking,
        "acceleration_windows": acceleration,
        "turning_windows": turning,
        "braking_rate_pct": rate(braking),
        "acceleration_rate_pct": rate(acceleration),
        "turning_rate_pct": rate(turning),
        "non_normal_event_rate_pct": rate(braking + acceleration + turning),
    }


def first_behavior(values):
    categories = values.dropna().astype("string").str.strip().str.upper()
    categories = categories[categories.ne("")]
    unique_categories = categories.unique().tolist()
    if len(unique_categories) > 1:
        raise ValueError(f"Conflicting behavior categories within a trip: {unique_categories}")
    return unique_categories[0] if unique_categories else "UNKNOWN"


def build_trip_rates(held_out, predictions, test_trip_ids):
    predictions = pd.Series(predictions, index=held_out.index, name="predicted_label")
    if not set(predictions.unique()).issubset(EVENT_NAMES):
        raise ValueError(f"Model returned unexpected event labels: {predictions.unique()}")

    rows = []
    for trip in test_trip_ids:
        trip_rows = held_out.loc[held_out["trip"].astype(str).eq(trip)]
        actual = pd.to_numeric(trip_rows["label"], errors="raise").astype(int)
        predicted = predictions.loc[trip_rows.index].astype(int)
        actual_stats = calculate_event_rates(actual)
        predicted_stats = calculate_event_rates(predicted)
        behavior = first_behavior(trip_rows["behavior"])

        row = {
            "trip": trip,
            "behavior": behavior,
            "total_windows": actual_stats["total_windows"],
        }
        for event, (actual_key, predicted_key) in EVENT_RATE_COLUMNS.items():
            rate_key = (
                "non_normal_event_rate_pct"
                if event == "Non-normal"
                else f"{event.lower()}_rate_pct"
            )
            actual_value = actual_stats[rate_key]
            predicted_value = predicted_stats[rate_key]
            row[actual_key] = actual_value
            row[predicted_key] = predicted_value
            error_name = (
                "non_normal_event_rate_error_pp"
                if event == "Non-normal"
                else f"{event.lower()}_rate_error_pp"
            )
            row[error_name] = predicted_value - actual_value

        for name in ["normal", "braking", "acceleration", "turning"]:
            row[f"actual_{name}_windows"] = actual_stats[f"{name}_windows"]
            row[f"predicted_{name}_windows"] = predicted_stats[f"{name}_windows"]
        rows.append(row)

    return pd.DataFrame(rows)


def build_summaries(trip_rates):
    known_behaviors = trip_rates[trip_rates["behavior"].isin(BEHAVIOR_CATEGORIES)]
    behavior_summary = (
        known_behaviors.groupby("behavior", sort=False)
        .agg(
            trip_count=("trip", "size"),
            actual_non_normal_event_rate_pct=(
                "actual_non_normal_event_rate_pct",
                "mean",
            ),
            predicted_non_normal_event_rate_pct=(
                "predicted_non_normal_event_rate_pct",
                "mean",
            ),
        )
        .reindex(BEHAVIOR_CATEGORIES)
        .reset_index()
    )
    behavior_summary["difference_pp"] = (
        behavior_summary["predicted_non_normal_event_rate_pct"]
        - behavior_summary["actual_non_normal_event_rate_pct"]
    )

    error_rows = []
    for event, (actual_column, predicted_column) in EVENT_RATE_COLUMNS.items():
        errors = trip_rates[predicted_column] - trip_rates[actual_column]
        error_rows.append(
            {
                "event": event,
                "trip_count": len(errors),
                "mean_actual_rate_pct": trip_rates[actual_column].mean(),
                "mean_predicted_rate_pct": trip_rates[predicted_column].mean(),
                "mean_error_pp": errors.mean(),
                "mean_absolute_error_pp": errors.abs().mean(),
                "root_mean_squared_error_pp": float(np.sqrt(np.mean(errors**2))),
                "overpredicted_trips": int((errors > 0).sum()),
                "underpredicted_trips": int((errors < 0).sum()),
                "equal_rate_trips": int((errors == 0).sum()),
            }
        )
    return behavior_summary, pd.DataFrame(error_rows)


def save_plots(behavior_summary, trip_rates, output_dir):
    colors = ["#4C78A8", "#E45756"]
    x = np.arange(len(BEHAVIOR_CATEGORIES))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))
    summary = behavior_summary.set_index("behavior").reindex(BEHAVIOR_CATEGORIES)
    axis.bar(
        x - width / 2,
        summary["actual_non_normal_event_rate_pct"],
        width,
        label="Actual",
        color=colors[0],
    )
    axis.bar(
        x + width / 2,
        summary["predicted_non_normal_event_rate_pct"],
        width,
        label="Predicted",
        color=colors[1],
    )
    axis.set_xticks(x)
    axis.set_xticklabels(BEHAVIOR_CATEGORIES)
    axis.set_xlabel("Driving behavior")
    axis.set_ylabel("Mean trip-level non-normal event rate (%)")
    axis.set_title("Actual vs Predicted Non-Normal Event Rates")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / OUTPUT_FILES["behavior_plot"], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    palette = {
        "NORMAL": "#4C78A8",
        "AGGRESSIVE": "#E45756",
        "DROWSY": "#72B7B2",
    }
    for behavior in BEHAVIOR_CATEGORIES:
        group = trip_rates[trip_rates["behavior"] == behavior]
        if not group.empty:
            axis.scatter(
                group["actual_non_normal_event_rate_pct"],
                group["predicted_non_normal_event_rate_pct"],
                label=behavior,
                color=palette[behavior],
                s=65,
                alpha=0.85,
            )

    maximum_rate = max(
        trip_rates["actual_non_normal_event_rate_pct"].max(),
        trip_rates["predicted_non_normal_event_rate_pct"].max(),
    )
    axis.plot([0, maximum_rate], [0, maximum_rate], linestyle="--", color="gray")
    axis.set_xlabel("Actual non-normal event rate (%)")
    axis.set_ylabel("Predicted non-normal event rate (%)")
    axis.set_title("Actual vs Predicted Rates for Held-Out Trips")
    axis.grid(alpha=0.25)
    axis.legend(title="Behavior")
    figure.tight_layout()
    figure.savefig(output_dir / OUTPUT_FILES["trip_plot"], dpi=150)
    plt.close(figure)


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "risk_analysis_outputs"
    dataset_path = project_dir / "uah_features.csv"
    metadata_path = output_dir / "reproduced_final_model_metadata.json"
    feature_list_path = output_dir / "reproduced_final_model_features.json"
    model_path = output_dir / "reproduced_final_event_classifier.joblib"

    for path in [dataset_path, metadata_path, feature_list_path, model_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = json.loads(feature_list_path.read_text(encoding="utf-8"))
    metadata_features = metadata.get("features")
    if not isinstance(feature_columns, list) or feature_columns != metadata_features:
        raise ValueError("Feature-list JSON does not match model metadata.")
    if len(feature_columns) != 29 or len(set(feature_columns)) != 29:
        raise ValueError("Expected 29 unique model feature names.")
    if FORBIDDEN_FEATURES.intersection(feature_columns):
        raise ValueError("Metadata, behavior, or target field found among model inputs.")

    expected_model_name = "reproduced_final_event_classifier.joblib"
    if metadata.get("model_file") != expected_model_name:
        raise ValueError("Metadata model_file does not match the required reproduced model.")
    expected_hash = metadata.get("model_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Missing or invalid model SHA-256 in metadata.")
    actual_hash = sha256_file(model_path)
    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            "Model SHA-256 mismatch; refusing to load the model. "
            f"Expected {expected_hash}, found {actual_hash}."
        )

    test_trip_ids = metadata.get("split", {}).get("test_trips")
    if not isinstance(test_trip_ids, list) or len(test_trip_ids) != 10:
        raise ValueError("Metadata must contain exactly 10 held-out test trips.")
    if len(set(test_trip_ids)) != len(test_trip_ids):
        raise ValueError("Held-out trip IDs in metadata are not unique.")

    data = pd.read_csv(dataset_path)
    required_columns = set(feature_columns).union(
        {"trip", "behavior", "label", "label_name"}
    )
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    numeric_features = [
        column
        for column in data.columns
        if column not in FORBIDDEN_FEATURES
        and column != "road"
        and pd.api.types.is_numeric_dtype(data[column])
    ]
    if numeric_features != feature_columns:
        raise ValueError(
            "Dataset numeric feature names/order differ from metadata: "
            f"{numeric_features}"
        )

    available_trips = set(data["trip"].dropna().astype(str))
    missing_test_trips = sorted(set(test_trip_ids).difference(available_trips))
    if missing_test_trips:
        raise ValueError(f"Held-out trips absent from dataset: {missing_test_trips}")
    held_out = data.loc[data["trip"].astype(str).isin(test_trip_ids)].copy()
    selected_trip_ids = set(held_out["trip"].astype(str).unique())
    if selected_trip_ids != set(test_trip_ids):
        raise ValueError("Filtered data does not exactly match metadata held-out trips.")

    labels = pd.to_numeric(held_out["label"], errors="raise")
    if labels.isna().any() or not labels.isin(EVENT_NAMES).all():
        raise ValueError("Held-out data contains missing or unexpected actual event labels.")
    label_names = (
        held_out.assign(_label=labels)
        .groupby("_label")["label_name"]
        .agg(lambda values: set(values.dropna().astype(str).str.strip().str.casefold()))
    )
    expected_label_names = {
        0: {"normal"},
        1: {"braking"},
        2: {"turning"},
        3: {"acceleration"},
    }
    if any(label_names.get(label, set()) != names for label, names in expected_label_names.items()):
        raise ValueError("label and label_name values do not match the expected event mapping.")

    inputs = held_out.loc[:, feature_columns]
    if not all(pd.api.types.is_numeric_dtype(inputs[column]) for column in feature_columns):
        raise ValueError("A model feature is not numeric.")
    if np.isinf(inputs.to_numpy(dtype=float)).any():
        raise ValueError("Infinite model input values found.")

    # Deserialize only after the checksum and input contract have been verified.
    model = joblib.load(model_path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Reproduced model does not provide predict().")
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != feature_columns:
            raise ValueError("Loaded model feature order differs from metadata.")
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_columns):
        raise ValueError("Loaded model expects a different number of features.")

    predictions = model.predict(inputs)
    trip_rates = build_trip_rates(held_out, predictions, test_trip_ids)
    if len(trip_rates) != 10:
        raise RuntimeError(f"Expected 10 trip rows, produced {len(trip_rates)}.")
    behavior_summary, error_summary = build_summaries(trip_rates)

    trip_rates.to_csv(output_dir / OUTPUT_FILES["trip_rates"], index=False)
    behavior_summary.to_csv(output_dir / OUTPUT_FILES["behavior_summary"], index=False)
    error_summary.to_csv(output_dir / OUTPUT_FILES["error_summary"], index=False)
    save_plots(behavior_summary, trip_rates, output_dir)

    behavior_rates = behavior_summary.set_index("behavior")
    normal_predicted = behavior_rates.loc["NORMAL", "predicted_non_normal_event_rate_pct"]
    aggressive_predicted = behavior_rates.loc[
        "AGGRESSIVE", "predicted_non_normal_event_rate_pct"
    ]
    normal_actual = behavior_rates.loc["NORMAL", "actual_non_normal_event_rate_pct"]
    aggressive_actual = behavior_rates.loc[
        "AGGRESSIVE", "actual_non_normal_event_rate_pct"
    ]
    preserves_trend = bool(aggressive_predicted > normal_predicted)
    turning_error = error_summary.loc[error_summary["event"] == "Turning"].iloc[0]

    analysis = {
        "model_sha256_verified": actual_hash,
        "held_out_trip_count": len(test_trip_ids),
        "held_out_row_count": len(held_out),
        "behavior_rate_summary": behavior_summary.to_dict(orient="records"),
        "trip_level_error_summary": error_summary.to_dict(orient="records"),
        "aggressive_vs_normal": {
            "actual_aggressive_mean_non_normal_rate_pct": float(aggressive_actual),
            "actual_normal_mean_non_normal_rate_pct": float(normal_actual),
            "predicted_aggressive_mean_non_normal_rate_pct": float(aggressive_predicted),
            "predicted_normal_mean_non_normal_rate_pct": float(normal_predicted),
            "predicted_trend_preserved": preserves_trend,
        },
        "turning_prediction": {
            "mean_actual_rate_pct": float(turning_error["mean_actual_rate_pct"]),
            "mean_predicted_rate_pct": float(turning_error["mean_predicted_rate_pct"]),
            "mean_error_pp": float(turning_error["mean_error_pp"]),
            "overpredicted_trips": int(turning_error["overpredicted_trips"]),
            "underpredicted_trips": int(turning_error["underpredicted_trips"]),
        },
        "risk_score_note": (
            "These are event-frequency estimates only; they are not validated risk scores."
        ),
    }
    (output_dir / OUTPUT_FILES["analysis"]).write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Verified model SHA-256: {actual_hash}")
    print(
        f"Analyzed only {len(test_trip_ids)} metadata-held-out trips "
        f"({len(held_out)} windows)."
    )
    print("\nActual and predicted non-normal event rates by behavior (mean trip rates):")
    print(
        behavior_summary.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    print("\nTrip-level event-rate error summary (predicted minus actual):")
    print(
        error_summary.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    turning_direction = (
        "overpredicted"
        if turning_error["mean_error_pp"] > 0
        else "underpredicted"
        if turning_error["mean_error_pp"] < 0
        else "matched"
    )
    print(
        f"\nTurning: {turning_direction} on average by "
        f"{abs(turning_error['mean_error_pp']):.2f} percentage points "
        f"({int(turning_error['overpredicted_trips'])} trips over, "
        f"{int(turning_error['underpredicted_trips'])} under)."
    )
    trend_phrase = "preserve" if preserves_trend else "do not preserve"
    print(
        f"Trend interpretation: Predictions {trend_phrase} the higher "
        "AGGRESSIVE-versus-NORMAL non-normal event-rate trend. "
        f"Actual means: AGGRESSIVE {aggressive_actual:.2f}% vs "
        f"NORMAL {normal_actual:.2f}%; predicted means: "
        f"AGGRESSIVE {aggressive_predicted:.2f}% vs NORMAL "
        f"{normal_predicted:.2f}%."
    )
    print(
        "These event-frequency estimates are descriptive and are not yet "
        "validated risk scores."
    )
    print(f"\nOutputs saved in: {output_dir}")
    for key, filename in OUTPUT_FILES.items():
        print(f"- {filename}")


if __name__ == "__main__":
    main()
