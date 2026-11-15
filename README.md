# Driver Behavior and Driving Risk Recognition from Smartphone Sensor Data

## Project Overview

This project focuses on recognizing driver behavior and driving risk using smartphone sensor data.

The project uses the UAH-DriveSet dataset, which contains driving data collected from smartphone sensors, GPS information, and annotated driving events.

The project now includes reproducible event-classifier training and held-out
evaluation, trip-level behavior analysis, an experimental risk indicator and
validation, and a Streamlit dashboard. The dashboard presents saved results
without running inference or retraining a model.

---

## Objectives

The main objectives of this project are:

- Collect and load smartphone sensor data.
- Process accelerometer and GPS data.
- Analyze sensor sampling rates and missing values.
- Preprocess the sensor data for analysis.
- Divide continuous sensor data into fixed-size time windows.
- Extract meaningful features from sensor signals.
- Identify driving events such as acceleration, braking, and turning.
- Visualize sensor signals and extracted features.
- Train and evaluate an event classifier using a trip-separated held-out test set.

---

## Dataset

### UAH-DriveSet

This project uses the UAH-DriveSet dataset.

The dataset contains driving data including:

- Accelerometer measurements
- GPS measurements
- Speed information
- Driving events
- Different driving behaviors

### Dataset Privacy / Repository Size

The raw UAH-DriveSet dataset is **not included in this GitHub repository** because of its large size.

The following files/directories are intentionally excluded:

```text
UAH-DRIVESET-v1/
UAH-DRIVESET-v1.zip
```

## Trip-Level Behavior Analysis (Phase 1)

From the project root, run:

```bash
python trip_behavior_analysis.py
```

The script reads `uah_features.csv` from the project root and writes trip-level
statistics, a behavior summary, and three plots to `risk_analysis_outputs/`.
Event counts and rates are calculated from `label_name`; behavior categories
are used only to group and compare trips. The summary averages trip-level
statistics, so trips are not weighted by their number of sensor windows.

## Reproduce the Final Event Classifier

Install the SMOTE dependency if needed, then run from the project root:

```bash
python -m pip install imbalanced-learn
python reproduce_final_model.py
```

The script reproduces the notebook's GroupShuffleSplit and trains only the
recorded HistGradientBoosting + SMOTE pipeline. It evaluates once on the
held-out trips and saves the separate model, feature list, configuration,
trip IDs, metrics, confusion matrix, and model SHA-256 under
`risk_analysis_outputs/`. It does not load or overwrite existing model files.

## Predicted Trip Event Analysis (Phase 2)

From the project root, run:

```bash
python predicted_trip_analysis.py
```

The script verifies the reproduced model checksum before loading it, then
analyzes only the 10 held-out trips listed in the model metadata. It writes
trip-level actual/predicted event rates, error and behavior summaries, and
comparison plots to `risk_analysis_outputs/`. These event-frequency estimates
are descriptive and are not validated risk scores.

## Experimental Risk Assessment (Phase 3)

From the project root, run:

```bash
python risk_assessment.py
```

The script verifies the reproduced model checksum, uses only its saved feature
list, and calibrates the experimental score and relative category cutoffs on
the 30 development trips. It then scores the 10 held-out trips and compares
their score distributions by behavior only after scoring is frozen. New
configuration, trip scores, validation summary, and plots are written to
`risk_analysis_outputs/`. Scores are experimental relative indicators, not
validated accident probabilities; event intensity/frequency does not mean an
event is inherently dangerous.

## Risk Indicator Validation (Phase 4)

From the project root, run:

```bash
python validate_risk_assessment.py
```

The script validates the combined and frequency-only indicators on the
metadata-listed held-out trips. It compares AGGRESSIVE with NORMAL and reports
DROWSY separately. Since Phase 3 did not define a binary behavior threshold,
threshold-dependent classification metrics and confusion matrices are marked
unavailable; Phase 3 relative risk categories are not repurposed as binary
thresholds. The validation remains exploratory and is not an accident-risk
probability evaluation.

## Streamlit Dashboard (Phase 5)

From the project root, install Streamlit if needed and start the dashboard:

```bash
python -m pip install streamlit
python -m streamlit run dashboard.py
```

Streamlit prints the local address (by default, `http://localhost:8501`).
The dashboard reads the saved CSV, JSON, and PNG analysis outputs in
`risk_analysis_outputs/`; it does not load the model, perform inference, or
retrain it.
