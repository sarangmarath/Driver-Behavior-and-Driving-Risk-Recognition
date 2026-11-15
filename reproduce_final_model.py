import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib


SEED = 42
LABEL_NAMES = ["Normal", "Braking", "Turning", "Acceleration"]
METADATA_COLUMNS = [
    "trip",
    "driver",
    "behavior",
    "road",
    "label",
    "label_name",
]
EXPECTED_FEATURES = [
    "ax_kf_mean",
    "ax_kf_std",
    "ax_kf_min",
    "ax_kf_max",
    "ay_kf_mean",
    "ay_kf_std",
    "ay_kf_min",
    "ay_kf_max",
    "az_kf_mean",
    "az_kf_std",
    "az_kf_min",
    "az_kf_max",
    "roll_mean",
    "roll_std",
    "roll_min",
    "roll_max",
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "yaw_mean",
    "yaw_std",
    "yaw_min",
    "yaw_max",
    "speed_mean",
    "speed_std",
    "speed_min",
    "speed_max",
    "acc_xy_mag_std",
]
NOTEBOOK_METRICS = {
    "accuracy": 0.9540,
    "macro_f1": 0.6532,
    "balanced_accuracy": 0.7522,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_path = project_dir / "uah_features.csv"
    output_dir = project_dir / "risk_analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    required_columns = set(METADATA_COLUMNS + EXPECTED_FEATURES)
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    numeric_features = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(data[column])
    ]
    if numeric_features != EXPECTED_FEATURES:
        raise ValueError(
            "Numeric feature names/order differ from the notebook contract: "
            f"{numeric_features}"
        )

    features = data[EXPECTED_FEATURES].copy()
    labels = data["label"].astype(int)
    trips = data["trip"].astype(str)
    expected_labels = {0, 1, 2, 3}
    if set(labels.unique()) != expected_labels:
        raise ValueError(f"Unexpected event labels: {sorted(labels.unique())}")
    if features.isna().any().any():
        raise ValueError("Missing sensor features found; notebook requires none.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite sensor features found; notebook requires none.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=SEED,
    )
    development_indices, test_indices = next(
        splitter.split(features, labels, groups=trips)
    )

    development_trips = trips.iloc[development_indices]
    test_trips = trips.iloc[test_indices]
    development_trip_ids = sorted(development_trips.unique().tolist())
    test_trip_ids = sorted(test_trips.unique().tolist())
    if set(development_trip_ids).intersection(test_trip_ids):
        raise RuntimeError("Trip leakage detected between development and test sets.")

    x_development = features.iloc[development_indices]
    y_development = labels.iloc[development_indices]
    x_test = features.iloc[test_indices]
    y_test = labels.iloc[test_indices]

    smote = SMOTE(k_neighbors=3, random_state=SEED)
    classifier = HistGradientBoostingClassifier(
        max_iter=100,
        random_state=SEED,
    )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", smote),
            ("model", classifier),
        ]
    )

    pipeline.fit(x_development, y_development)
    predictions = pipeline.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3])
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "per_class": {
            name: report[name]
            for name in LABEL_NAMES
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "held_out_rows": int(len(y_test)),
    }

    model_path = output_dir / "reproduced_final_event_classifier.joblib"
    joblib.dump(pipeline, model_path)
    model_sha256 = sha256_file(model_path)

    (output_dir / "reproduced_final_model_features.json").write_text(
        json.dumps(EXPECTED_FEATURES, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "source_notebook": "notebooks/train_models.ipynb",
        "recorded_selection": {
            "classifier": "HistGradientBoostingClassifier",
            "imbalance_strategy": "SMOTE",
            "development_group_cv": "GroupKFold(n_splits=3)",
            "selection_evidence": "Notebook cell 7 output ranks Hist Gradient Boosting + smote first.",
        },
        "split": {
            "method": "GroupShuffleSplit",
            "n_splits": 1,
            "test_size": 0.25,
            "random_state": SEED,
            "development_rows": int(len(development_indices)),
            "test_rows": int(len(test_indices)),
            "development_trips": development_trip_ids,
            "test_trips": test_trip_ids,
        },
        "features": EXPECTED_FEATURES,
        "excluded_model_inputs": METADATA_COLUMNS,
        "target": "label",
        "pipeline": {
            "imputer": pipeline.named_steps["imputer"].get_params(deep=False),
            "scaler": pipeline.named_steps["scaler"].get_params(deep=False),
            "smote": pipeline.named_steps["smote"].get_params(deep=False),
            "classifier": pipeline.named_steps["model"].get_params(deep=False),
        },
        "training_scope": "Fit only on development rows; no held-out test data used for fitting or selection.",
        "software_versions": {
            package: version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "imbalanced-learn",
                "joblib",
                "matplotlib",
            ]
        },
        "model_file": model_path.name,
        "model_sha256": model_sha256,
        "notebook_recorded_metrics": NOTEBOOK_METRICS,
    }
    (output_dir / "reproduced_final_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    evaluation = {
        "new_run": metrics,
        "notebook_recorded_metrics": NOTEBOOK_METRICS,
        "difference_from_notebook": {
            key: metrics[key] - recorded
            for key, recorded in NOTEBOOK_METRICS.items()
        },
        "comparison_note": (
            "Notebook metrics are rounded to four decimal places. The notebook "
            "does not record package versions, so differences can reflect "
            "software-version/default implementation changes."
        ),
    }
    (output_dir / "reproduced_final_model_evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )

    matrix_frame = pd.DataFrame(
        matrix,
        index=pd.Index(LABEL_NAMES, name="actual"),
        columns=pd.Index(LABEL_NAMES, name="predicted"),
    )
    matrix_frame.to_csv(output_dir / "reproduced_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(len(LABEL_NAMES)),
        xticklabels=LABEL_NAMES,
        yticks=range(len(LABEL_NAMES)),
        yticklabels=LABEL_NAMES,
        xlabel="Predicted event",
        ylabel="Actual event",
        title="Reproduced Classifier — Held-Out Test Trips",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "reproduced_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Model: {model_path}")
    print(f"SHA-256: {model_sha256}")
    print(
        f"Split: {len(development_indices)} development rows / "
        f"{len(development_trip_ids)} trips; {len(test_indices)} held-out rows / "
        f"{len(test_trip_ids)} trips"
    )
    print("\nHeld-out test metrics:")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print("\nNotebook metric comparison (new minus recorded):")
    for name, difference in evaluation["difference_from_notebook"].items():
        print(f"{name}: {difference:+.4f}")
    print(f"\nAll new outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
