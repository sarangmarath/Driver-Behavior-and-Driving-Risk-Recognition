import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BEHAVIOR_CATEGORIES = ["NORMAL", "AGGRESSIVE", "DROWSY"]
EVENT_CLASSES = {
    "Braking": 1,
    "Acceleration": 3,
    "Turning": 2,
}
FORBIDDEN_FEATURES = {
    "behavior",
    "trip",
    "driver",
    "road",
    "label",
    "label_name",
}
SCORING_WEIGHTS = {
    "event_type": {
        "Braking": 1 / 3,
        "Acceleration": 1 / 3,
        "Turning": 1 / 3,
    },
    "frequency": 0.5,
    "intensity": 0.5,
}
QUANTILE = 0.90
DEVELOPMENT_SCORE_QUANTILES = [0.33, 0.67]
OUTPUT_FILES = {
    "scores": "risk_scores_by_trip.csv",
    "config": "risk_scoring_config.json",
    "validation": "risk_validation_summary.json",
    "distribution_plot": "risk_score_distribution_by_behavior.png",
    "comparison_plot": "frequency_only_vs_combined_indicator.png",
}
INTENSITY_SOURCE_COLUMNS = [
    "ax_kf_min",
    "ax_kf_max",
    "ay_kf_min",
    "ay_kf_max",
    "yaw_std",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q90(values, description):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No development observations available for {description}.")
    scale = float(np.quantile(values, QUANTILE))
    if scale <= 0:
        raise ValueError(
            f"Development 90th-percentile scale for {description} is zero; "
            "risk indicator calibration is not possible."
        )
    return scale


def check_longitudinal_direction(development, development_trip_ids):
    paired = development.loc[
        development["label"].isin([1, 3]), ["trip", "label", "ax_kf_mean"]
    ].copy()
    trip_class_medians = paired.pivot_table(
        index="trip",
        columns="label",
        values="ax_kf_mean",
        aggfunc="median",
    )
    trip_class_medians = trip_class_medians.dropna(subset=[1, 3])
    differences = trip_class_medians[3] - trip_class_medians[1]
    signs = np.sign(differences.to_numpy(dtype=float))
    global_acceleration_median = float(
        paired.loc[paired["label"] == 3, "ax_kf_mean"].median()
    )
    global_braking_median = float(
        paired.loc[paired["label"] == 1, "ax_kf_mean"].median()
    )
    global_difference = global_acceleration_median - global_braking_median

    direction_reliable = (
        len(differences) >= 10
        and len(signs) > 0
        and global_difference != 0
        and np.all(signs == np.sign(global_difference))
    )
    polarity = int(np.sign(global_difference)) if direction_reliable else None
    diagnostic = {
        "feature_extraction_evidence": (
            "The feature notebook identifies ax as longitudinal but does not "
            "document forward-positive versus forward-negative polarity."
        ),
        "development_only_check": (
            "Compare per-trip median ax_kf_mean for actual Acceleration and "
            "Braking windows; this check is used only to assess axis direction."
        ),
        "development_trip_count": len(development_trip_ids),
        "trips_with_both_event_classes": int(len(differences)),
        "acceleration_median_ax_kf_mean": global_acceleration_median,
        "braking_median_ax_kf_mean": global_braking_median,
        "global_median_difference_acceleration_minus_braking": float(
            global_difference
        ),
        "trips_with_positive_difference": int((differences > 0).sum()),
        "trips_with_negative_difference": int((differences < 0).sum()),
        "trips_with_zero_difference": int((differences == 0).sum()),
        "reliable_direction": bool(direction_reliable),
        "polarity": polarity,
        "fallback": (
            None
            if direction_reliable
            else "Use max(abs(ax_kf_min), abs(ax_kf_max)) for both event types; "
            "this is a longitudinal-excursion proxy, not signed deceleration "
            "or acceleration intensity."
        ),
    }
    return polarity, diagnostic


def prepare_sensor_intensities(features, development_medians, polarity):
    sensors = features.loc[:, INTENSITY_SOURCE_COLUMNS].copy()
    sensors = sensors.fillna(development_medians)
    if sensors.isna().any().any():
        raise ValueError("An intensity sensor could not be imputed from development data.")
    if np.isinf(sensors.to_numpy(dtype=float)).any():
        raise ValueError("Infinite sensor values found in intensity inputs.")

    intensity = pd.DataFrame(index=sensors.index)
    if polarity is None:
        longitudinal_excursion = np.maximum(
            sensors["ax_kf_min"].abs(),
            sensors["ax_kf_max"].abs(),
        )
        intensity["braking_intensity"] = longitudinal_excursion
        intensity["acceleration_intensity"] = longitudinal_excursion
    else:
        intensity["braking_intensity"] = (
            -polarity * sensors["ax_kf_min"]
        ).clip(lower=0)
        intensity["acceleration_intensity"] = (
            polarity * sensors["ax_kf_max"]
        ).clip(lower=0)
    intensity["lateral_excursion"] = np.maximum(
        sensors["ay_kf_min"].abs(),
        sensors["ay_kf_max"].abs(),
    )
    intensity["yaw_variability"] = sensors["yaw_std"].abs()
    return intensity


def fit_calibration(
    development_features,
    development_predictions,
    development_trip_ids,
    polarity,
):
    predictions = pd.Series(
        development_predictions,
        index=development_features.index,
        name="predicted_label",
    )
    event_frequencies = {event: [] for event in EVENT_CLASSES}
    for trip in development_trip_ids:
        trip_indices = development_features.index[
            development_features["trip"].astype(str).eq(trip)
        ]
        trip_predictions = predictions.loc[trip_indices]
        total = len(trip_indices)
        if total == 0:
            raise ValueError(f"Development trip has no rows: {trip}")
        for event, class_id in EVENT_CLASSES.items():
            rate = int((trip_predictions == class_id).sum()) / total
            if rate > 0:
                event_frequencies[event].append(rate)

    frequency_scales = {
        event: q90(rates, f"nonzero development {event} trip frequency")
        for event, rates in event_frequencies.items()
    }

    intensity_columns = development_features.loc[:, INTENSITY_SOURCE_COLUMNS]
    development_medians = intensity_columns.median()
    if development_medians.isna().any():
        missing = development_medians[development_medians.isna()].index.tolist()
        raise ValueError(f"Cannot impute intensity columns from development data: {missing}")
    development_intensities = prepare_sensor_intensities(
        development_features, development_medians, polarity
    )

    intensity_scales = {}
    calibration_event_windows = {}
    for event, class_id in EVENT_CLASSES.items():
        mask = predictions == class_id
        calibration_event_windows[event] = int(mask.sum())
        if not mask.any():
            raise ValueError(
                f"The reproduced classifier predicted no development {event} "
                "windows; its intensity scale cannot be calibrated."
            )
        if event == "Turning":
            intensity_scales[event] = {
                column: q90(
                    development_intensities.loc[mask, column],
                    f"development {event} {column}",
                )
                for column in ["lateral_excursion", "yaw_variability"]
            }
        else:
            intensity_scales[event] = {
                "event_intensity": q90(
                    development_intensities.loc[mask, f"{event.lower()}_intensity"],
                    f"development {event} intensity",
                )
            }

    return {
        "frequency_scales": frequency_scales,
        "frequency_positive_trip_counts": {
            event: len(rates) for event, rates in event_frequencies.items()
        },
        "intensity_scales": intensity_scales,
        "intensity_development_event_windows": calibration_event_windows,
        "development_sensor_medians": {
            name: float(value) for name, value in development_medians.items()
        },
    }


def score_trips(
    trip_features,
    predictions,
    trip_ids,
    calibration,
    development_medians,
    polarity,
):
    predictions = pd.Series(
        predictions,
        index=trip_features.index,
        name="predicted_label",
    )
    intensities = prepare_sensor_intensities(
        trip_features, development_medians, polarity
    )
    frequency_scales = calibration["frequency_scales"]
    intensity_scales = calibration["intensity_scales"]
    rows = []

    for trip in trip_ids:
        indices = trip_features.index[trip_features["trip"].astype(str).eq(trip)]
        if len(indices) == 0:
            raise ValueError(f"Trip has no rows to score: {trip}")
        trip_predictions = predictions.loc[indices]
        total_windows = len(indices)
        frequency_only_components = []
        combined_components = []
        row = {"trip": trip, "total_windows": total_windows}

        for event, class_id in EVENT_CLASSES.items():
            mask = trip_predictions == class_id
            event_count = int(mask.sum())
            frequency = event_count / total_windows
            normalized_frequency = float(
                np.clip(frequency / frequency_scales[event], 0, 1)
            )
            if event_count == 0:
                normalized_intensity = 0.0
                combined_component = 0.0
            elif event == "Turning":
                scales = intensity_scales[event]
                lateral = np.clip(
                    intensities.loc[indices[mask.to_numpy()], "lateral_excursion"]
                    / scales["lateral_excursion"],
                    0,
                    1,
                )
                yaw = np.clip(
                    intensities.loc[indices[mask.to_numpy()], "yaw_variability"]
                    / scales["yaw_variability"],
                    0,
                    1,
                )
                normalized_intensity = float((lateral.mean() + yaw.mean()) / 2)
                combined_component = normalized_frequency * (
                    0.5 + 0.5 * normalized_intensity
                )
            else:
                scale = intensity_scales[event]["event_intensity"]
                event_intensity = intensities.loc[
                    indices[mask.to_numpy()], f"{event.lower()}_intensity"
                ]
                normalized_intensity = float(
                    np.clip(event_intensity / scale, 0, 1).mean()
                )
                combined_component = normalized_frequency * (
                    0.5 + 0.5 * normalized_intensity
                )

            frequency_only_components.append(normalized_frequency)
            combined_components.append(combined_component)
            key = event.lower()
            row[f"predicted_{key}_windows"] = event_count
            row[f"predicted_{key}_frequency"] = frequency
            row[f"normalized_{key}_frequency"] = normalized_frequency
            row[f"normalized_{key}_intensity"] = normalized_intensity
            row[f"{key}_combined_component"] = combined_component

        row["frequency_only_indicator"] = (
            100 * sum(frequency_only_components) / len(EVENT_CLASSES)
        )
        row["combined_risk_indicator"] = (
            100 * sum(combined_components) / len(EVENT_CLASSES)
        )
        rows.append(row)

    scored = pd.DataFrame(rows)
    if not np.isfinite(
        scored[["frequency_only_indicator", "combined_risk_indicator"]].to_numpy()
    ).all():
        raise ValueError("Non-finite risk indicators generated.")
    score_columns = ["frequency_only_indicator", "combined_risk_indicator"]
    if ((scored[score_columns] < 0) | (scored[score_columns] > 100)).any().any():
        raise ValueError("Risk indicator fell outside the required [0, 100] bounds.")
    if polarity is None:
        scored["longitudinal_intensity_method"] = "absolute_excursion_proxy"
    else:
        scored["longitudinal_intensity_method"] = "development_verified_direction"
    return scored


def risk_category(scores, low_cutoff, high_cutoff):
    if scores <= low_cutoff:
        return "Low"
    if scores <= high_cutoff:
        return "Moderate"
    return "High"


def behavior_by_trip(rows):
    result = {}
    for trip, trip_rows in rows.groupby("trip", sort=False):
        values = (
            trip_rows["behavior"]
            .dropna()
            .astype("string")
            .str.strip()
            .str.upper()
        )
        values = values[values.ne("")]
        unique = values.unique().tolist()
        if len(unique) > 1:
            raise ValueError(f"Conflicting behavior values within trip {trip}: {unique}")
        result[str(trip)] = unique[0] if unique else "UNKNOWN"
    return result


def heldout_event_rate_check(held_out, heldout_predictions):
    predicted = pd.Series(
        heldout_predictions,
        index=held_out.index,
        name="predicted_label",
    )
    rows = []
    for trip, trip_rows in held_out.groupby("trip", sort=False):
        actual_labels = pd.to_numeric(trip_rows["label"], errors="raise").astype(int)
        predicted_labels = predicted.loc[trip_rows.index].astype(int)
        count = len(trip_rows)
        for event, class_id in EVENT_CLASSES.items():
            actual_rate = float((actual_labels == class_id).sum() / count)
            predicted_rate = float((predicted_labels == class_id).sum() / count)
            rows.append(
                {
                    "trip": str(trip),
                    "event": event,
                    "actual_rate": actual_rate,
                    "predicted_rate": predicted_rate,
                    "error": predicted_rate - actual_rate,
                }
            )
    rates = pd.DataFrame(rows)
    summary = {}
    for event, group in rates.groupby("event", sort=False):
        mean_error = float(group["error"].mean())
        summary[event] = {
            "mean_actual_trip_rate_pct": float(group["actual_rate"].mean() * 100),
            "mean_predicted_trip_rate_pct": float(
                group["predicted_rate"].mean() * 100
            ),
            "mean_error_percentage_points": mean_error * 100,
            "overpredicted_trips": int((group["error"] > 0).sum()),
            "underpredicted_trips": int((group["error"] < 0).sum()),
            "equal_rate_trips": int((group["error"] == 0).sum()),
        }
    return summary


def save_graphs(scored_heldout, output_dir):
    colors = {
        "NORMAL": "#4C78A8",
        "AGGRESSIVE": "#E45756",
        "DROWSY": "#72B7B2",
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    data = [
        scored_heldout.loc[
            scored_heldout["behavior"] == behavior,
            "combined_risk_indicator",
        ].to_numpy()
        for behavior in BEHAVIOR_CATEGORIES
    ]
    present = [
        (behavior, values)
        for behavior, values in zip(BEHAVIOR_CATEGORIES, data)
        if len(values)
    ]
    if present:
        axis.boxplot(
            [values for _, values in present],
            positions=np.arange(1, len(present) + 1),
            widths=0.55,
            showfliers=False,
        )
        for position, (behavior, values) in enumerate(present, start=1):
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else [0]
            axis.scatter(
                position + jitter,
                values,
                color=colors.get(behavior, "gray"),
                label=behavior,
                zorder=3,
            )
        handles, labels = axis.get_legend_handles_labels()
        unique_handles = dict(zip(labels, handles))
        axis.legend(unique_handles.values(), unique_handles.keys(), title="Behavior")
        axis.set_xticks(
            np.arange(1, len(present) + 1),
            [behavior for behavior, _ in present],
        )
    axis.set_ylim(0, 100)
    axis.set_ylabel("Experimental risk indicator (0–100)")
    axis.set_title("Held-Out Risk Indicator Distribution by Behavior")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / OUTPUT_FILES["distribution_plot"], dpi=150)
    plt.close(figure)

    ordered = scored_heldout.reset_index(drop=True)
    positions = np.arange(len(ordered))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        positions - width / 2,
        ordered["frequency_only_indicator"],
        width,
        label="Frequency-only baseline",
        color="#9ECAE1",
    )
    axis.bar(
        positions + width / 2,
        ordered["combined_risk_indicator"],
        width,
        label="Frequency + intensity",
        color="#F28E2B",
    )
    labels = [str(trip)[-18:] for trip in ordered["trip"]]
    axis.set_xticks(positions, labels, rotation=40, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Experimental indicator (0–100)")
    axis.set_xlabel("Held-out trip ID (suffix)")
    axis.set_title("Frequency-Only vs Combined Indicator on Held-Out Trips")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / OUTPUT_FILES["comparison_plot"], dpi=150)
    plt.close(figure)


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "risk_analysis_outputs"
    dataset_path = project_dir / "uah_features.csv"
    metadata_path = output_dir / "reproduced_final_model_metadata.json"
    features_path = output_dir / "reproduced_final_model_features.json"
    model_path = output_dir / "reproduced_final_event_classifier.joblib"

    for path in [dataset_path, metadata_path, features_path, model_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
    if feature_columns != metadata.get("features") or len(feature_columns) != 29:
        raise ValueError("Saved feature list does not match model metadata.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature list contains duplicate names.")
    if FORBIDDEN_FEATURES.intersection(feature_columns):
        raise ValueError("A forbidden metadata or target column appears in the features.")
    if metadata.get("model_file") != model_path.name:
        raise ValueError("Metadata model_file does not match the reproduced model.")
    expected_hash = metadata.get("model_sha256")
    actual_hash = sha256_file(model_path)
    if not isinstance(expected_hash, str) or actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            "Reproduced model checksum mismatch; refusing to deserialize it. "
            f"Expected {expected_hash}, found {actual_hash}."
        )

    split_metadata = metadata.get("split", {})
    development_trip_ids = split_metadata.get("development_trips")
    heldout_trip_ids = split_metadata.get("test_trips")
    if (
        not isinstance(development_trip_ids, list)
        or not isinstance(heldout_trip_ids, list)
        or len(development_trip_ids) != 30
        or len(heldout_trip_ids) != 10
        or set(development_trip_ids).intersection(heldout_trip_ids)
    ):
        raise ValueError("Saved model metadata does not contain the verified 30/10 split.")

    data = pd.read_csv(dataset_path)
    required = set(feature_columns).union({"trip", "label", "behavior"})
    missing_columns = sorted(required.difference(data.columns))
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
        raise ValueError("Dataset numeric feature order differs from saved feature list.")
    data["trip"] = data["trip"].astype(str)
    present_trips = set(data["trip"].unique())
    if (set(development_trip_ids) | set(heldout_trip_ids)) != present_trips:
        raise ValueError("Dataset trip IDs do not exactly match the saved 30/10 split.")

    development = data.loc[data["trip"].isin(development_trip_ids)].copy()
    held_out = data.loc[data["trip"].isin(heldout_trip_ids)].copy()
    if (
        len(development) != split_metadata.get("development_rows")
        or len(held_out) != split_metadata.get("test_rows")
        or development["trip"].nunique() != 30
        or held_out["trip"].nunique() != 10
    ):
        raise ValueError("Dataset rows/trips differ from the saved training/test split.")

    development_model_inputs = development.loc[:, feature_columns]
    heldout_model_inputs = held_out.loc[:, feature_columns]
    development_scoring_data = development.loc[:, ["trip", *feature_columns]]
    heldout_scoring_data = held_out.loc[:, ["trip", *feature_columns]]
    for features in [development_model_inputs, heldout_model_inputs]:
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise ValueError("Infinite model input values found.")

    # Verify the axis sign using development labels only. This check is not
    # used in scoring calibration; the observed inconsistency selects abs().
    polarity, axis_diagnostic = check_longitudinal_direction(
        development, development_trip_ids
    )

    model = joblib.load(model_path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Reproduced model does not expose predict().")
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != feature_columns:
            raise ValueError("Loaded model feature order differs from metadata.")
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_columns):
        raise ValueError("Loaded model expects a different feature count.")

    development_predictions = model.predict(development_model_inputs)
    heldout_predictions = model.predict(heldout_model_inputs)
    if not np.array_equal(
        development_predictions, model.predict(development_model_inputs)
    ) or not np.array_equal(
        heldout_predictions, model.predict(heldout_model_inputs)
    ):
        raise RuntimeError("Model predictions are not deterministic.")
    allowed_predictions = set(EVENT_CLASSES.values()) | {0}
    if not set(np.unique(development_predictions)).issubset(allowed_predictions):
        raise ValueError("Unexpected event class predicted for development trips.")
    if not set(np.unique(heldout_predictions)).issubset(allowed_predictions):
        raise ValueError("Unexpected event class predicted for held-out trips.")

    calibration = fit_calibration(
        development_scoring_data,
        development_predictions,
        development_trip_ids,
        polarity,
    )
    development_medians = pd.Series(calibration["development_sensor_medians"])
    development_scores = score_trips(
        development_scoring_data,
        development_predictions,
        development_trip_ids,
        calibration,
        development_medians,
        polarity,
    )
    development_scores_repeat = score_trips(
        development_scoring_data,
        development_predictions,
        development_trip_ids,
        calibration,
        development_medians,
        polarity,
    )
    score_pair = ["frequency_only_indicator", "combined_risk_indicator"]
    if not np.array_equal(
        development_scores[score_pair].to_numpy(),
        development_scores_repeat[score_pair].to_numpy(),
    ):
        raise RuntimeError("Development scoring is not deterministic.")

    low_cutoff, high_cutoff = np.quantile(
        development_scores["combined_risk_indicator"],
        DEVELOPMENT_SCORE_QUANTILES,
    ).tolist()
    config = {
        "model_file": model_path.name,
        "model_sha256_verified": actual_hash,
        "feature_list": feature_columns,
        "split": {
            "development_trips": development_trip_ids,
            "heldout_trips": heldout_trip_ids,
            "development_rows": int(len(development)),
            "heldout_rows": int(len(held_out)),
        },
        "axis_direction_diagnostic": axis_diagnostic,
        "longitudinal_intensity": (
            "absolute longitudinal excursion proxy for both braking and "
            "acceleration; the feature code marks ax longitudinal but does "
            "not establish sign, and development per-trip medians disagree."
            if polarity is None
            else "signed direction verified on development trips"
        ),
        "calibration": {
            "source": (
                "Reproduced classifier predictions and sensor features from "
                "development trips only; behavior and event labels are not "
                "used for calibration."
            ),
            "quantile": QUANTILE,
            "frequency_scales": calibration["frequency_scales"],
            "nonzero_frequency_trip_counts": calibration[
                "frequency_positive_trip_counts"
            ],
            "intensity_scales": calibration["intensity_scales"],
            "predicted_event_window_counts": calibration[
                "intensity_development_event_windows"
            ],
            "sensor_medians": calibration["development_sensor_medians"],
            "development_score_quantiles": DEVELOPMENT_SCORE_QUANTILES,
            "low_cutoff": float(low_cutoff),
            "high_cutoff": float(high_cutoff),
            "development_scores_are_in_sample": True,
            "category_ties": bool(np.isclose(low_cutoff, high_cutoff)),
        },
        "formula": (
            "R_t = (100/3) * sum_k(f_tk * (0.5 + 0.5*I_tk)); "
            "frequency-only baseline = (100/3) * sum_k(f_tk)"
        ),
        "weights": SCORING_WEIGHTS,
        "intensity_definitions": {
            "Braking": (
                "max(0, -polarity*ax_kf_min) if direction verified; otherwise "
                "max(abs(ax_kf_min), abs(ax_kf_max))"
            ),
            "Acceleration": (
                "max(0, polarity*ax_kf_max) if direction verified; otherwise "
                "max(abs(ax_kf_min), abs(ax_kf_max))"
            ),
            "Turning": (
                "0.5*clip(lateral_excursion/Q90_dev,0,1) + "
                "0.5*clip(abs(yaw_std)/Q90_dev,0,1)"
            ),
            "lateral_excursion": "max(abs(ay_kf_min), abs(ay_kf_max))",
        },
        "outlier_handling": (
            "Normalize by development-only 90th percentiles and clip each "
            "normalized frequency/intensity to [0,1]."
        ),
        "zero_predicted_events": (
            "Set class frequency, intensity, and combined class contribution to zero."
        ),
        "relative_categories": {
            "Low": f"score <= {float(low_cutoff)}",
            "Moderate": f"{float(low_cutoff)} < score <= {float(high_cutoff)}",
            "High": f"score > {float(high_cutoff)}",
            "cutoffs_fit_on": "30 development trip scores only",
        },
        "limitations": [
            "Experimental relative behavior indicator, not accident probability.",
            "A predicted braking, acceleration, or turning event is not automatically dangerous.",
            "The classifier's development predictions are in-sample.",
            "Risk categories are relative to the development data.",
            "Feature summaries do not capture raw-event duration or jerk.",
        ],
    }

    config_path = output_dir / OUTPUT_FILES["config"]
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        saved_config["model_sha256_verified"] != actual_hash
        or saved_config["feature_list"] != feature_columns
        or saved_config["split"]["heldout_trips"] != heldout_trip_ids
    ):
        raise RuntimeError("Saved risk configuration failed consistency checks.")

    # Freeze all calibration and category cutoffs before examining held-out behavior.
    heldout_scores = score_trips(
        heldout_scoring_data,
        heldout_predictions,
        heldout_trip_ids,
        calibration,
        development_medians,
        polarity,
    )
    heldout_scores_repeat = score_trips(
        heldout_scoring_data,
        heldout_predictions,
        heldout_trip_ids,
        calibration,
        development_medians,
        polarity,
    )
    if not np.array_equal(
        heldout_scores[score_pair].to_numpy(),
        heldout_scores_repeat[score_pair].to_numpy(),
    ):
        raise RuntimeError("Held-out risk scoring is not deterministic.")
    heldout_scores["relative_risk_category"] = heldout_scores[
        "combined_risk_indicator"
    ].map(lambda score: risk_category(score, low_cutoff, high_cutoff))

    # Behavior is read only after the risk scores and thresholds are frozen.
    heldout_behaviors = behavior_by_trip(held_out)
    heldout_scores["behavior"] = heldout_scores["trip"].map(heldout_behaviors)
    unexpected_behaviors = sorted(
        set(heldout_scores["behavior"]).difference(BEHAVIOR_CATEGORIES)
    )

    behavior_summary = (
        heldout_scores.loc[
            heldout_scores["behavior"].isin(BEHAVIOR_CATEGORIES)
        ]
        .groupby("behavior", sort=False)
        .agg(
            trip_count=("trip", "size"),
            combined_mean=("combined_risk_indicator", "mean"),
            combined_median=("combined_risk_indicator", "median"),
            frequency_only_mean=("frequency_only_indicator", "mean"),
            high_category_trips=(
                "relative_risk_category",
                lambda values: int((values == "High").sum()),
            ),
        )
        .reindex(BEHAVIOR_CATEGORIES)
        .reset_index()
    )

    event_overprediction = heldout_event_rate_check(held_out, heldout_predictions)
    risk_means = behavior_summary.set_index("behavior")["combined_mean"]
    aggressive_mean = risk_means.get("AGGRESSIVE")
    normal_mean = risk_means.get("NORMAL")
    if pd.notna(aggressive_mean) and pd.notna(normal_mean):
        aggressive_vs_normal = {
            "aggressive_mean_combined_indicator": float(aggressive_mean),
            "normal_mean_combined_indicator": float(normal_mean),
            "difference_points": float(aggressive_mean - normal_mean),
            "aggressive_mean_higher": bool(aggressive_mean > normal_mean),
        }
    else:
        aggressive_vs_normal = {
            "warning": "NORMAL or AGGRESSIVE is absent from the held-out trips."
        }

    warnings = [
        "This is an experimental relative driving-behavior indicator, not a validated accident probability.",
        "A predicted braking, acceleration, or turning event is not automatically dangerous.",
        "Development predictions are in-sample and are used only for development calibration.",
        "Low/Moderate/High thresholds are relative to development scores and are not universal risk cutoffs.",
        "Behavior comparisons are descriptive on only 10 held-out trips and were not used to tune scoring.",
    ]
    if polarity is None:
        warnings.append(
            "Longitudinal polarity was not reliably established; absolute excursion "
            "is used for braking and acceleration intensity."
        )
    if unexpected_behaviors:
        warnings.append(f"Unexpected held-out behavior values excluded from category comparison: {unexpected_behaviors}")
    if config["calibration"]["category_ties"]:
        warnings.append(
            "Development score quantile cutoffs are tied; one or more relative "
            "risk categories may contain no trips."
        )
    if any(
        summary["mean_error_percentage_points"] > 0
        for summary in event_overprediction.values()
    ):
        warnings.append(
            "Held-out event predictions show mean overprediction for one or more "
            f"event classes: {event_overprediction}."
        )

    validation = {
        "calibration_frozen_before_heldout_behavior_evaluation": True,
        "heldout_trip_count": int(len(heldout_scores)),
        "heldout_row_count": int(len(held_out)),
        "heldout_scores_by_behavior": behavior_summary.to_dict(orient="records"),
        "aggressive_vs_normal": aggressive_vs_normal,
        "frequency_only_vs_combined": {
            "mean_frequency_only_indicator": float(
                heldout_scores["frequency_only_indicator"].mean()
            ),
            "mean_combined_indicator": float(
                heldout_scores["combined_risk_indicator"].mean()
            ),
            "mean_combined_minus_frequency_only": float(
                (
                    heldout_scores["combined_risk_indicator"]
                    - heldout_scores["frequency_only_indicator"]
                ).mean()
            ),
            "per_behavior": behavior_summary[
                [
                    "behavior",
                    "trip_count",
                    "frequency_only_mean",
                    "combined_mean",
                ]
            ].to_dict(orient="records"),
        },
        "known_classifier_event_overprediction": event_overprediction,
        "warnings": warnings,
        "validation_checks": {
            "model_sha256_matches_metadata": True,
            "29_features_match_saved_order": True,
            "development_test_trip_ids_disjoint": True,
            "30_development_and_10_heldout_trips": True,
            "expected_split_row_counts": True,
            "risk_scores_finite_and_in_0_100": True,
            "development_and_heldout_predictions_deterministic": True,
            "configuration_matches_metadata": True,
            "axis_direction_verified_or_fallback_documented": True,
            "behavior_used_only_after_scoring": True,
        },
    }

    scores_path = output_dir / OUTPUT_FILES["scores"]
    validation_path = output_dir / OUTPUT_FILES["validation"]
    heldout_scores.to_csv(scores_path, index=False)
    validation_path.write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_graphs(heldout_scores, output_dir)

    print(f"Verified reproduced model SHA-256: {actual_hash}")
    print(
        f"Split verified: {len(development_trip_ids)} development trips, "
        f"{len(heldout_trip_ids)} held-out trips; {len(held_out)} held-out windows."
    )
    print("\nDevelopment-only calibration:")
    print(json.dumps(config["calibration"], indent=2))
    print("\nAxis-direction diagnostic:")
    print(json.dumps(axis_diagnostic, indent=2))
    print("\nHeld-out trip scores:")
    print(
        heldout_scores[
            [
                "trip",
                "behavior",
                "frequency_only_indicator",
                "combined_risk_indicator",
                "relative_risk_category",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )
    print("\nHeld-out means by behavior:")
    print(
        behavior_summary.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    print("\nFrequency-only vs combined mean indicators:")
    print(
        f"Frequency-only: {validation['frequency_only_vs_combined']['mean_frequency_only_indicator']:.2f}; "
        f"combined: {validation['frequency_only_vs_combined']['mean_combined_indicator']:.2f}; "
        f"difference: {validation['frequency_only_vs_combined']['mean_combined_minus_frequency_only']:.2f}"
    )
    print("\nKnown event prediction bias on held-out trips:")
    for event, summary in event_overprediction.items():
        print(
            f"{event}: mean error {summary['mean_error_percentage_points']:+.2f} pp "
            f"(over {summary['overpredicted_trips']}, "
            f"under {summary['underpredicted_trips']})"
        )
    print("\nWarnings:")
    for warning in warnings:
        print(f"- {warning}")
    print("\nNew Phase 3 outputs:")
    for filename in OUTPUT_FILES.values():
        print(f"- {filename}")


if __name__ == "__main__":
    main()
