import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "risk_analysis_outputs"
HELDOUT_EVENTS_FILE = OUTPUT_DIR / "predicted_trip_event_rates_heldout.csv"
RISK_SCORES_FILE = OUTPUT_DIR / "risk_scores_by_trip.csv"
MODEL_EVALUATION_FILE = OUTPUT_DIR / "reproduced_final_model_evaluation.json"
CONFUSION_MATRIX_FILE = OUTPUT_DIR / "reproduced_confusion_matrix.csv"
CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "reproduced_confusion_matrix.png"
PHASE4_VALIDATION_FILE = OUTPUT_DIR / "risk_indicator_validation_summary.json"
PHASE4_COMPARISON_FILE = OUTPUT_DIR / "risk_indicator_method_comparison.csv"
RISK_DISTRIBUTION_PLOT = OUTPUT_DIR / "risk_score_distribution_by_behavior.png"
RISK_COMPARISON_PLOT = OUTPUT_DIR / "frequency_only_vs_combined_indicator.png"
ROC_PLOT = OUTPUT_DIR / "risk_indicator_roc_curves.png"
HELDOUT_COMPARISON_PLOT = OUTPUT_DIR / "risk_indicator_heldout_score_comparison.png"


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def main():
    st.set_page_config(
        page_title="Driving Risk Assessment",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Driver Behavior & Driving Risk Recognition")
    st.caption("Smartphone-sensor event recognition and experimental trip-level analysis")

    missing_files = [
        path.name
        for path in [
            HELDOUT_EVENTS_FILE,
            RISK_SCORES_FILE,
            MODEL_EVALUATION_FILE,
            CONFUSION_MATRIX_FILE,
            CONFUSION_MATRIX_PLOT,
            PHASE4_VALIDATION_FILE,
            PHASE4_COMPARISON_FILE,
            RISK_DISTRIBUTION_PLOT,
            RISK_COMPARISON_PLOT,
            ROC_PLOT,
            HELDOUT_COMPARISON_PLOT,
        ]
        if not path.is_file()
    ]
    if missing_files:
        st.error(
            "Required saved analysis outputs are missing from "
            f"`risk_analysis_outputs/`: {', '.join(missing_files)}"
        )
        st.info(
            "Run the Phase 1–4 analysis scripts first. This dashboard does not "
            "run model inference or retrain models."
        )
        st.stop()

    event_rates = load_csv(str(HELDOUT_EVENTS_FILE))
    risk_scores = load_csv(str(RISK_SCORES_FILE))
    confusion_matrix = load_csv(str(CONFUSION_MATRIX_FILE))
    phase4_comparison = load_csv(str(PHASE4_COMPARISON_FILE))
    model_evaluation = load_json(str(MODEL_EVALUATION_FILE))
    phase4_validation = load_json(str(PHASE4_VALIDATION_FILE))

    if (
        event_rates["trip"].duplicated().any()
        or risk_scores["trip"].duplicated().any()
        or set(event_rates["trip"]) != set(risk_scores["trip"])
        or len(event_rates) != 10
    ):
        st.error("Saved Phase 2 and Phase 3 files do not match the 10 held-out trips.")
        st.stop()

    st.markdown(
        """
        This project uses smartphone sensor features to recognize window-level
        driving events—Normal, Braking, Acceleration, and Turning. A separate
        experimental indicator combines predicted event frequency with sensor
        intensity summaries to compare trip-level behavior.
        """
    )
    st.warning(
        "Risk indicators and Low/Moderate/High categories are experimental and "
        "relative to the development data. They are **not validated accident "
        "probabilities**, and a braking, acceleration, or turning event is not "
        "automatically dangerous."
    )

    sidebar = st.sidebar
    sidebar.header("Held-out trip")
    sidebar.caption("All trip-level views below use the original 10 held-out trips.")
    risk_by_trip = risk_scores.set_index("trip")
    trip_ids = sorted(event_rates["trip"].tolist())
    selected_trip = sidebar.selectbox(
        "Select a trip",
        trip_ids,
        format_func=lambda trip: (
            f"{trip} · {risk_by_trip.loc[trip, 'behavior']}"
        ),
    )

    selected_rates = event_rates.loc[event_rates["trip"] == selected_trip].iloc[0]
    selected_risk = risk_by_trip.loc[selected_trip]
    if selected_rates["behavior"] != selected_risk["behavior"]:
        st.error("Behavior fields differ between the saved event and risk outputs.")
        st.stop()

    st.subheader("Selected held-out trip")
    st.write(f"**Trip:** `{selected_trip}`")
    st.write(
        f"**Behavior group:** {selected_risk['behavior']} · "
        f"**Windows:** {int(selected_rates['total_windows']):,}"
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Frequency-only indicator",
        f"{selected_risk['frequency_only_indicator']:.2f} / 100",
    )
    metric_columns[1].metric(
        "Combined indicator",
        f"{selected_risk['combined_risk_indicator']:.2f} / 100",
    )
    metric_columns[2].metric(
        "Relative category",
        str(selected_risk["relative_risk_category"]),
    )
    st.caption(
        "The category is based on development-set score quantiles; it is a "
        "relative indicator, not a universal risk threshold."
    )

    st.subheader("Actual vs predicted event rates")
    selected_event_rates = pd.DataFrame(
        {
            "Event": ["Braking", "Acceleration", "Turning", "Non-normal"],
            "Actual (%)": [
                selected_rates["actual_braking_rate_pct"],
                selected_rates["actual_acceleration_rate_pct"],
                selected_rates["actual_turning_rate_pct"],
                selected_rates["actual_non_normal_event_rate_pct"],
            ],
            "Predicted (%)": [
                selected_rates["predicted_braking_rate_pct"],
                selected_rates["predicted_acceleration_rate_pct"],
                selected_rates["predicted_turning_rate_pct"],
                selected_rates["predicted_non_normal_event_rate_pct"],
            ],
        }
    ).set_index("Event")
    st.bar_chart(selected_event_rates, height=320)
    st.caption("Rates are the percentage of sensor windows in this trip.")

    st.divider()
    st.subheader("Trip-level risk indicator distributions")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.image(
            str(RISK_DISTRIBUTION_PLOT),
            caption="Combined indicator distributions by behavior (held-out trips)",
            width="stretch",
        )
    with chart_columns[1]:
        st.image(
            str(RISK_COMPARISON_PLOT),
            caption="Frequency-only and combined scores for held-out trips",
            width="stretch",
        )

    st.subheader("Held-out event-classifier evaluation")
    st.caption(
        "Metrics below are from the original held-out set (8,042 windows), "
        "not development data."
    )
    model_metrics = model_evaluation["new_run"]
    score_columns = st.columns(3)
    score_columns[0].metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
    score_columns[1].metric("Macro-F1", f"{model_metrics['macro_f1']:.4f}")
    score_columns[2].metric(
        "Balanced accuracy",
        f"{model_metrics['balanced_accuracy']:.4f}",
    )
    confusion_columns = st.columns([1, 1])
    with confusion_columns[0]:
        st.image(
            str(CONFUSION_MATRIX_PLOT),
            caption="Confusion matrix — rows are actual, columns are predicted",
            width="stretch",
        )
    with confusion_columns[1]:
        st.markdown("**Confusion matrix values**")
        st.dataframe(
            confusion_matrix.set_index("actual"),
            width="stretch",
        )

    st.subheader("Phase 4: AGGRESSIVE vs NORMAL discrimination")
    binary_counts = phase4_validation["binary_behavior_definition"]
    st.caption(
        "DROWSY trips are excluded from this binary analysis and shown only "
        "descriptively. The validation has "
        f"{binary_counts['aggressive_trip_count']} AGGRESSIVE and "
        f"{binary_counts['normal_trip_count']} NORMAL trips "
        f"({binary_counts['aggressive_normal_pair_count']} cross-group pairs)."
    )
    st.warning(
        "ROC-AUC is a ranking statistic—not accuracy or a claim of 100% "
        "real-world accuracy. With only five trips in the binary comparison, "
        "these results are highly uncertain."
    )
    method_table = phase4_comparison[
        [
            "indicator",
            "roc_auc",
            "mean_aggressive_minus_normal",
            "median_aggressive_minus_normal",
            "correctly_ranked_pairs",
            "aggressive_normal_pairs",
            "exact_two_sided_auc_permutation_p",
        ]
    ].rename(
        columns={
            "indicator": "Indicator",
            "roc_auc": "ROC-AUC",
            "mean_aggressive_minus_normal": "Mean score difference",
            "median_aggressive_minus_normal": "Median score difference",
            "correctly_ranked_pairs": "Correctly ranked pairs",
            "aggressive_normal_pairs": "Total pairs",
            "exact_two_sided_auc_permutation_p": "Exact permutation p",
        }
    )
    st.dataframe(method_table, hide_index=True, width="stretch")
    st.image(
        str(ROC_PLOT),
        caption="Held-out ROC curves; DROWSY excluded",
        width="stretch",
    )
    st.image(
        str(HELDOUT_COMPARISON_PLOT),
        caption="Frequency-only and combined indicator values by held-out behavior",
        width="stretch",
    )

    threshold_info = phase4_validation["binary_threshold"]
    st.info(
        "Balanced accuracy, precision, recall, F1, and binary confusion matrices "
        f"are {threshold_info['status']}: {threshold_info['reason']}"
    )
    st.caption(
        "Phase 3 Low/Moderate/High cutoffs describe relative development-set "
        "score bands; they are not binary AGGRESSIVE-vs-NORMAL decision thresholds."
    )

    with st.expander("Interpretation and limitations"):
        st.markdown(
            """
            - Phase 1–4 summaries use saved analysis files; the dashboard does not
              load a classifier or run predictions.
            - Risk indicators are experimental behavior summaries, not validated
              accident-risk probabilities.
            - ROC-AUC describes ranking on this small held-out sample. It is not
              accuracy, calibration, or real-world safety performance.
            - Threshold-dependent binary metrics are not shown because no
              development-fixed binary decision threshold was saved.
            - Development predictions used for risk calibration are in-sample;
              held-out results are presented separately.
            """
        )


if __name__ == "__main__":
    main()
