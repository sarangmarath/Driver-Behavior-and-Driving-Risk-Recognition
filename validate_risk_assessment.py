from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


BEHAVIOR_CATEGORIES = ["NORMAL", "AGGRESSIVE", "DROWSY"]
INDICATORS = {
    "combined": "combined_risk_indicator",
    "frequency_only": "frequency_only_indicator",
}
OUTPUT_FILES = {
    "validation": "risk_indicator_validation_summary.json",
    "comparison": "risk_indicator_method_comparison.csv",
    "behavior_summary": "risk_indicator_behavior_summary.csv",
    "confusion_matrices": "risk_indicator_confusion_matrices.json",
    "roc_plot": "risk_indicator_roc_curves.png",
    "score_plot": "risk_indicator_heldout_score_comparison.png",
}


def exact_auc_permutation_p_value(labels, scores, observed_auc):
    positive_count = int(np.sum(labels == 1))
    sample_count = len(labels)
    if positive_count == 0 or positive_count == sample_count:
        return None, 0

    aucs = []
    for positive_indices in combinations(range(sample_count), positive_count):
        permuted = np.zeros(sample_count, dtype=int)
        permuted[list(positive_indices)] = 1
        aucs.append(roc_auc_score(permuted, scores))

    observed_distance = abs(observed_auc - 0.5)
    extreme = sum(abs(value - 0.5) >= observed_distance - 1e-12 for value in aucs)
    return extreme / len(aucs), len(aucs)


def pairwise_ranking(labels, scores):
    aggressive_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    pair_count = len(aggressive_scores) * len(normal_scores)
    if pair_count == 0:
        return {
            "correct_pairs": 0,
            "ties": 0,
            "incorrect_pairs": 0,
            "total_pairs": pair_count,
            "ranking_percentage": None,
            "tie_adjusted_ranking_percentage": None,
        }

    wins = 0
    ties = 0
    for aggressive_score in aggressive_scores:
        for normal_score in normal_scores:
            if aggressive_score > normal_score:
                wins += 1
            elif np.isclose(aggressive_score, normal_score, rtol=0, atol=1e-12):
                ties += 1
    losses = pair_count - wins - ties
    return {
        "correct_pairs": wins,
        "ties": ties,
        "incorrect_pairs": losses,
        "total_pairs": pair_count,
        "ranking_percentage": 100 * wins / pair_count,
        "tie_adjusted_ranking_percentage": 100 * (wins + 0.5 * ties) / pair_count,
    }


def binary_metrics(labels, scores):
    result = {
        "roc_auc": None,
        "exact_two_sided_auc_permutation_p": None,
        "auc_permutations": 0,
        "pairwise_ranking": pairwise_ranking(labels, scores),
        "threshold_dependent_metrics": {
            "status": "unavailable",
            "binary_threshold": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": None,
            "reason": (
                "Phase 3 has no development-fixed binary AGGRESSIVE-vs-NORMAL "
                "threshold. Its Low/Moderate/High relative cutoffs are not "
                "binary decision thresholds; no held-out threshold was selected."
            ),
        },
        "roc_curve": None,
    }
    if len(np.unique(labels)) == 2:
        score = float(roc_auc_score(labels, scores))
        p_value, permutation_count = exact_auc_permutation_p_value(
            labels, scores, score
        )
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        result.update(
            {
                "roc_auc": score,
                "exact_two_sided_auc_permutation_p": p_value,
                "auc_permutations": permutation_count,
                "roc_curve": {
                    "false_positive_rate": false_positive_rate.tolist(),
                    "true_positive_rate": true_positive_rate.tolist(),
                    "curve_auc": float(auc(false_positive_rate, true_positive_rate)),
                },
            }
        )
    return result


def save_roc_plot(method_results, output_dir):
    figure, axis = plt.subplots(figsize=(7, 6))
    colors = {"combined": "#E45756", "frequency_only": "#4C78A8"}
    for method, result in method_results.items():
        curve = result["roc_curve"]
        if curve is not None:
            axis.plot(
                curve["false_positive_rate"],
                curve["true_positive_rate"],
                marker="o",
                label=f"{method.replace('_', ' ').title()} (AUC={result['roc_auc']:.2f})",
                color=colors[method],
            )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Held-Out AGGRESSIVE vs NORMAL (DROWSY excluded)",
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / OUTPUT_FILES["roc_plot"], dpi=150)
    plt.close(figure)


def save_score_plot(data, output_dir):
    colors = {"frequency_only": "#4C78A8", "combined": "#E45756"}
    positions = {category: index for index, category in enumerate(BEHAVIOR_CATEGORIES)}
    offsets = {"frequency_only": -0.10, "combined": 0.10}
    figure, axis = plt.subplots(figsize=(9, 6))

    for behavior in BEHAVIOR_CATEGORIES:
        group = data[data["behavior"] == behavior]
        center = positions[behavior]
        for _, row in group.iterrows():
            for method, column in INDICATORS.items():
                axis.scatter(
                    center + offsets[method],
                    row[column],
                    color=colors[method],
                    marker="o",
                    s=55,
                    label=method.replace("_", " ").title(),
                    zorder=3,
                )
            axis.plot(
                [center + offsets["frequency_only"], center + offsets["combined"]],
                [row[INDICATORS["frequency_only"]], row[INDICATORS["combined"]]],
                color="gray",
                alpha=0.4,
                linewidth=1,
            )

    handles, labels = axis.get_legend_handles_labels()
    unique_handles = dict(zip(labels, handles))
    axis.legend(unique_handles.values(), unique_handles.keys())
    axis.set_xticks(
        range(len(BEHAVIOR_CATEGORIES)),
        BEHAVIOR_CATEGORIES,
    )
    axis.set_ylim(0, 105)
    axis.set_ylabel("Relative indicator (0–100)")
    axis.set_title("Held-Out Scores by Behavior and Indicator")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / OUTPUT_FILES["score_plot"], dpi=150)
    plt.close(figure)


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "risk_analysis_outputs"
    scores_path = output_dir / "risk_scores_by_trip.csv"
    config_path = output_dir / "risk_scoring_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [scores_path, config_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 3 file not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    scores = pd.read_csv(scores_path)
    required_columns = {
        "trip",
        "behavior",
        "combined_risk_indicator",
        "frequency_only_indicator",
        "relative_risk_category",
    }
    missing_columns = sorted(required_columns.difference(scores.columns))
    if missing_columns:
        raise ValueError(f"Phase 3 scores are missing columns: {missing_columns}")

    heldout_trip_ids = config.get("split", {}).get("heldout_trips")
    if not isinstance(heldout_trip_ids, list) or len(heldout_trip_ids) != 10:
        raise ValueError("Phase 3 config does not contain exactly 10 held-out trip IDs.")
    if scores["trip"].duplicated().any() or set(scores["trip"]) != set(heldout_trip_ids):
        raise ValueError("Phase 3 score rows do not exactly match the held-out trip IDs.")
    if len(scores) != 10:
        raise ValueError(f"Expected 10 held-out score rows, found {len(scores)}.")

    for column in INDICATORS.values():
        values = pd.to_numeric(scores[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
            raise ValueError(f"Invalid value in Phase 3 score column {column}.")
        scores[column] = values

    scores["behavior"] = scores["behavior"].astype("string").str.strip().str.upper()
    unexpected = sorted(set(scores["behavior"].dropna()).difference(BEHAVIOR_CATEGORIES))
    if unexpected:
        raise ValueError(f"Unexpected behavior categories in held-out scores: {unexpected}")

    # Keep DROWSY for descriptive reporting only; the binary evaluation is Aggressive vs Normal.
    binary = scores[scores["behavior"].isin(["AGGRESSIVE", "NORMAL"])].copy()
    group_counts = binary["behavior"].value_counts().to_dict()
    aggressive_count = int(group_counts.get("AGGRESSIVE", 0))
    normal_count = int(group_counts.get("NORMAL", 0))
    drowsy = scores[scores["behavior"] == "DROWSY"]
    if aggressive_count == 0 or normal_count == 0:
        raise ValueError("Both AGGRESSIVE and NORMAL held-out trips are required for binary AUC.")

    labels = (binary["behavior"] == "AGGRESSIVE").astype(int).to_numpy()
    method_results = {}
    comparison_rows = []
    for method, column in INDICATORS.items():
        values = binary[column].to_numpy(dtype=float)
        metrics = binary_metrics(labels, values)
        method_results[method] = metrics
        aggressive = binary.loc[binary["behavior"] == "AGGRESSIVE", column]
        normal = binary.loc[binary["behavior"] == "NORMAL", column]
        pairwise = metrics["pairwise_ranking"]
        comparison_rows.append(
            {
                "indicator": method,
                "aggressive_trip_count": aggressive_count,
                "normal_trip_count": normal_count,
                "aggressive_normal_pairs": pairwise["total_pairs"],
                "roc_auc": metrics["roc_auc"],
                "exact_two_sided_auc_permutation_p": metrics[
                    "exact_two_sided_auc_permutation_p"
                ],
                "mean_aggressive_minus_normal": float(
                    aggressive.mean() - normal.mean()
                ),
                "median_aggressive_minus_normal": float(
                    aggressive.median() - normal.median()
                ),
                "correctly_ranked_pairs": pairwise["correct_pairs"],
                "tied_pairs": pairwise["ties"],
                "incorrectly_ranked_pairs": pairwise["incorrect_pairs"],
                "pairwise_ranking_pct_ties_half_credit": pairwise[
                    "tie_adjusted_ranking_percentage"
                ],
                "threshold_dependent_metrics": "unavailable",
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    behavior_summary = (
        scores.groupby("behavior", sort=False)
        .agg(
            trip_count=("trip", "size"),
            combined_mean=("combined_risk_indicator", "mean"),
            combined_median=("combined_risk_indicator", "median"),
            frequency_only_mean=("frequency_only_indicator", "mean"),
            frequency_only_median=("frequency_only_indicator", "median"),
        )
        .reindex(BEHAVIOR_CATEGORIES)
        .reset_index()
    )
    drowsy_summary = behavior_summary.loc[
        behavior_summary["behavior"] == "DROWSY"
    ].to_dict(orient="records")

    combined_auc = method_results["combined"]["roc_auc"]
    frequency_auc = method_results["frequency_only"]["roc_auc"]
    combined_rank = method_results["combined"]["pairwise_ranking"][
        "tie_adjusted_ranking_percentage"
    ]
    frequency_rank = method_results["frequency_only"]["pairwise_ranking"][
        "tie_adjusted_ranking_percentage"
    ]
    if combined_auc > frequency_auc:
        discrimination_result = "combined indicator has higher held-out ROC-AUC"
    elif combined_auc < frequency_auc:
        discrimination_result = "frequency-only baseline has higher held-out ROC-AUC"
    elif combined_rank > frequency_rank:
        discrimination_result = (
            "ROC-AUC is tied; combined indicator has a higher pairwise ranking percentage"
        )
    elif combined_rank < frequency_rank:
        discrimination_result = (
            "ROC-AUC is tied; frequency-only baseline has a higher pairwise ranking percentage"
        )
    else:
        discrimination_result = (
            "no observed discrimination improvement; ROC-AUC and pairwise ranking are tied"
        )

    relative_cutoffs = config.get("relative_categories", {})
    confusion_status = {
        method: {
            "status": "unavailable",
            "matrix": None,
            "threshold": None,
            "reason": (
                "No development-fixed binary AGGRESSIVE-vs-NORMAL threshold is "
                "stored in Phase 3. The Phase 3 Low/Moderate/High score cutoffs "
                "are relative categories, not binary decision thresholds. "
                "No threshold was selected from held-out labels."
            ),
        }
        for method in INDICATORS
    }
    validation = {
        "scope": "Original 10 held-out trips only.",
        "binary_behavior_definition": {
            "positive": "AGGRESSIVE",
            "negative": "NORMAL",
            "excluded_from_binary_analysis": "DROWSY",
            "aggressive_trip_count": aggressive_count,
            "normal_trip_count": normal_count,
            "drowsy_trip_count_descriptive_only": int(len(drowsy)),
            "aggressive_normal_pair_count": aggressive_count * normal_count,
        },
        "phase3_relative_categories_not_binary_thresholds": relative_cutoffs,
        "binary_threshold": {
            "value": None,
            "status": "not available",
            "reason": (
                "Phase 3 stored development-only Low/Moderate/High quantile "
                "cutoffs, but no separately constructed binary threshold."
            ),
            "threshold_dependent_metrics": (
                "Balanced accuracy, precision, recall, F1, and confusion "
                "matrices are unavailable; no held-out threshold was chosen."
            ),
        },
        "indicator_results": method_results,
        "drowsy_descriptive_summary": drowsy_summary,
        "comparison": {
            "combined_minus_frequency_only_roc_auc": float(
                combined_auc - frequency_auc
            ),
            "combined_minus_frequency_only_pairwise_ranking_percentage_points": float(
                combined_rank - frequency_rank
            ),
            "result": discrimination_result,
            "interpretation_note": (
                "Numerically smaller or larger scores do not establish better "
                "discrimination; compare rank-based metrics."
            ),
        },
        "small_sample_note": (
            f"The binary comparison has {aggressive_count} AGGRESSIVE and "
            f"{normal_count} NORMAL trips, only {aggressive_count * normal_count} "
            "cross-group pairs. Exact two-sided label-permutation p-values are "
            "reported for AUC; estimates are highly uncertain and descriptive."
        ),
        "risk_interpretation_limit": (
            "These are experimental driving-behavior indicators, not validated "
            "accident-risk probabilities."
        ),
    }

    pd.DataFrame(comparison_rows).to_csv(
        output_dir / OUTPUT_FILES["comparison"], index=False
    )
    behavior_summary.to_csv(
        output_dir / OUTPUT_FILES["behavior_summary"], index=False
    )
    (output_dir / OUTPUT_FILES["confusion_matrices"]).write_text(
        json.dumps(confusion_status, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / OUTPUT_FILES["validation"]).write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_roc_plot(method_results, output_dir)
    save_score_plot(scores, output_dir)

    print("Held-out binary comparison: AGGRESSIVE vs NORMAL (DROWSY excluded)")
    print(
        f"Trips: AGGRESSIVE={aggressive_count}, NORMAL={normal_count}, "
        f"DROWSY={len(drowsy)} (descriptive only); "
        f"AGGRESSIVE-NORMAL pairs={aggressive_count * normal_count}."
    )
    print("\nIndicator comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBehavior-group score summary (DROWSY descriptive only):")
    print(
        behavior_summary.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    print("\nThreshold metrics: unavailable; no development-fixed binary threshold exists.")
    print(
        "Phase 3 relative Low/Moderate/High cutoffs are not used as binary "
        "AGGRESSIVE-vs-NORMAL thresholds."
    )
    print(f"\nConclusion: {discrimination_result}.")
    print(validation["small_sample_note"])
    print(validation["risk_interpretation_limit"])
    print("\nNew Phase 4 outputs:")
    for filename in OUTPUT_FILES.values():
        print(f"- {filename}")


if __name__ == "__main__":
    main()
