from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BEHAVIOR_CATEGORIES = ["NORMAL", "AGGRESSIVE", "DROWSY"]
EVENT_LABELS = {
    "normal": "normal_windows",
    "braking": "braking_windows",
    "acceleration": "acceleration_windows",
    "turning": "turning_windows",
}
SUMMARY_METRICS = [
    "total_windows",
    "normal_windows",
    "braking_windows",
    "acceleration_windows",
    "turning_windows",
    "non_normal_event_rate_pct",
    "braking_rate_pct",
    "acceleration_rate_pct",
    "turning_rate_pct",
    "average_speed",
    "average_acc_xy_mag_std",
]


def first_non_missing(values):
    non_missing = values.dropna()
    return non_missing.iloc[0] if not non_missing.empty else pd.NA


def build_trip_statistics(data):
    normalized_labels = data["label_name"].astype("string").str.strip().str.casefold()
    normalized_labels = normalized_labels.mask(normalized_labels.eq(""))
    data = data.copy()
    data["_event_label"] = normalized_labels
    data["_behavior_category"] = (
        data["behavior"].astype("string").str.strip().str.upper()
    )

    unknown_labels = data.loc[
        data["_event_label"].notna()
        & ~data["_event_label"].isin(EVENT_LABELS),
        "_event_label",
    ]
    missing_labels = int(data["_event_label"].isna().sum())
    if missing_labels:
        print(f"Data note: {missing_labels} rows have a missing/blank label_name.")
    if not unknown_labels.empty:
        counts = unknown_labels.value_counts().sort_index().to_dict()
        print(f"Data note: unrecognized label_name values (excluded from event counts): {counts}")

    for column in ["trip", "driver", "road"]:
        missing_count = int(data[column].isna().sum())
        if missing_count:
            handling = (
                "missing trip IDs are grouped together."
                if column == "trip"
                else "missing metadata is retained in the trip output."
            )
            print(
                f"Data note: {missing_count} rows have a missing {column}; {handling}"
            )

    expected_behaviors = set(BEHAVIOR_CATEGORIES)
    unexpected_behaviors = data.loc[
        data["_behavior_category"].notna()
        & ~data["_behavior_category"].isin(expected_behaviors),
        "_behavior_category",
    ]
    missing_behaviors = int(data["_behavior_category"].isna().sum())
    if missing_behaviors:
        print(f"Data note: {missing_behaviors} rows have a missing/blank behavior.")
    if not unexpected_behaviors.empty:
        counts = unexpected_behaviors.value_counts().sort_index().to_dict()
        print(
            "Data note: unexpected behavior values are retained in trip statistics "
            f"but excluded from the three-category summary: {counts}"
        )

    numeric_columns = ["speed_mean", "acc_xy_mag_std"]
    for column in numeric_columns:
        original_non_missing = data[column].notna()
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_count = int((original_non_missing & converted.isna()).sum())
        missing_count = int(data[column].isna().sum())
        if missing_count or invalid_count:
            print(
                f"Data note: {column} has {missing_count} missing and "
                f"{invalid_count} non-numeric value(s); averages skip these values."
            )
        data[column] = converted

    trip_rows = []
    for trip, trip_data in data.groupby("trip", dropna=False, sort=True):
        total_windows = len(trip_data)
        event_counts = trip_data["_event_label"].value_counts()
        normal = int(event_counts.get("normal", 0))
        braking = int(event_counts.get("braking", 0))
        acceleration = int(event_counts.get("acceleration", 0))
        turning = int(event_counts.get("turning", 0))

        def percent(count):
            return count / total_windows * 100 if total_windows else float("nan")

        trip_rows.append(
            {
                "trip": trip,
                "driver_id": first_non_missing(trip_data["driver"]),
                "behavior": first_non_missing(trip_data["_behavior_category"]),
                "road_type": first_non_missing(trip_data["road"]),
                "total_windows": total_windows,
                "normal_windows": normal,
                "braking_windows": braking,
                "acceleration_windows": acceleration,
                "turning_windows": turning,
                "non_normal_event_rate_pct": percent(
                    braking + acceleration + turning
                ),
                "braking_rate_pct": percent(braking),
                "acceleration_rate_pct": percent(acceleration),
                "turning_rate_pct": percent(turning),
                "average_speed": trip_data["speed_mean"].mean(),
                "average_acc_xy_mag_std": trip_data["acc_xy_mag_std"].mean(),
            }
        )

    trip_statistics = pd.DataFrame(trip_rows)
    for metadata_column, source_column in [
        ("driver_id", "driver"),
        ("behavior", "_behavior_category"),
        ("road_type", "road"),
    ]:
        inconsistent = data.groupby("trip", dropna=False)[source_column].nunique(
            dropna=True
        )
        inconsistent = inconsistent[inconsistent > 1]
        if not inconsistent.empty:
            print(
                f"Data note: {len(inconsistent)} trip(s) have conflicting "
                f"{metadata_column} values; the first non-missing value is used."
            )

    return trip_statistics


def build_behavior_summary(trip_statistics):
    known_trips = trip_statistics[
        trip_statistics["behavior"].isin(BEHAVIOR_CATEGORIES)
    ]
    means = known_trips.groupby("behavior")[SUMMARY_METRICS].mean()
    summary = pd.DataFrame(index=BEHAVIOR_CATEGORIES)
    summary.index.name = "behavior"
    summary["trip_count"] = known_trips.groupby("behavior").size()
    summary = summary.join(means).reset_index()
    summary["trip_count"] = summary["trip_count"].fillna(0).astype(int)
    return summary


def save_plots(trip_statistics, behavior_summary, output_dir):
    categories = BEHAVIOR_CATEGORIES
    colors = ["#4C78A8", "#E45756", "#72B7B2"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        behavior_summary["behavior"],
        behavior_summary["non_normal_event_rate_pct"],
        color=colors,
    )
    ax.set_title("Average Non-Normal Event Rate by Driving Behavior")
    ax.set_xlabel("Driving behavior")
    ax.set_ylabel("Mean trip-level event rate (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "average_non_normal_event_rate_by_behavior.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = list(range(len(categories)))
    bar_width = 0.24
    rate_series = [
        ("braking_rate_pct", "Braking"),
        ("acceleration_rate_pct", "Acceleration"),
        ("turning_rate_pct", "Turning"),
    ]
    for offset, (column, label) in enumerate(rate_series):
        values = (
            behavior_summary.set_index("behavior")[column]
            .reindex(categories)
            .to_numpy()
        )
        ax.bar(
            [position + (offset - 1) * bar_width for position in x_positions],
            values,
            width=bar_width,
            label=label,
        )
    ax.set_title("Event Rates by Driving Behavior")
    ax.set_xlabel("Driving behavior")
    ax.set_ylabel("Mean trip-level event rate (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "event_rates_by_behavior.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    box_data = [
        trip_statistics.loc[
            trip_statistics["behavior"] == category,
            "non_normal_event_rate_pct",
        ].dropna()
        for category in categories
    ]
    non_empty = [
        (category, values)
        for category, values in zip(categories, box_data)
        if not values.empty
    ]
    if non_empty:
        ax.boxplot(
            [values for _, values in non_empty],
            patch_artist=True,
        )
        ax.set_xticks(range(1, len(non_empty) + 1))
        ax.set_xticklabels([category for category, _ in non_empty])
    else:
        ax.text(0.5, 0.5, "No trips with recognized behavior categories", ha="center")
        ax.set_xticks([])
    ax.set_title("Trip-Level Non-Normal Event Rate Distribution")
    ax.set_xlabel("Driving behavior")
    ax.set_ylabel("Non-normal event rate (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "non_normal_event_rate_distribution_by_trip.png", dpi=150)
    plt.close(fig)


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_path = project_dir / "uah_features.csv"
    output_dir = project_dir / "risk_analysis_outputs"

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    required_columns = {
        "trip",
        "driver",
        "behavior",
        "road",
        "label_name",
        "speed_mean",
        "acc_xy_mag_std",
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    trip_statistics = build_trip_statistics(data)
    behavior_summary = build_behavior_summary(trip_statistics)
    output_dir.mkdir(parents=True, exist_ok=True)

    trip_statistics.to_csv(output_dir / "trip_level_statistics.csv", index=False)
    behavior_summary.to_csv(output_dir / "behavior_summary.csv", index=False)
    save_plots(trip_statistics, behavior_summary, output_dir)

    print(f"\nTrip-level statistics saved: {output_dir / 'trip_level_statistics.csv'}")
    print(f"Behavior summary saved: {output_dir / 'behavior_summary.csv'}")
    print(f"Plots saved in: {output_dir}")
    print("\nBehavior summary (mean of trip-level statistics):")
    print(behavior_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    rates = behavior_summary.set_index("behavior")["non_normal_event_rate_pct"]
    normal_rate = rates.get("NORMAL")
    aggressive_rate = rates.get("AGGRESSIVE")
    print("\nInterpretation:")
    if pd.notna(normal_rate) and pd.notna(aggressive_rate):
        difference = aggressive_rate - normal_rate
        if difference > 0:
            conclusion = "Yes"
            explanation = f"{difference:.2f} percentage points higher"
        elif difference < 0:
            conclusion = "No"
            explanation = f"{abs(difference):.2f} percentage points lower"
        else:
            conclusion = "No"
            explanation = "equal to"
        print(
            f"{conclusion} - AGGRESSIVE trips average {aggressive_rate:.2f}% "
            f"non-normal events versus {normal_rate:.2f}% for NORMAL trips "
            f"({explanation}, using the mean of trip-level rates)."
        )
    else:
        print(
            "Unable to compare AGGRESSIVE and NORMAL trips: at least one "
            "category has no trips with a calculable rate."
        )


if __name__ == "__main__":
    main()
