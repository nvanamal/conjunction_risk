#!/usr/bin/env python3
"""Single-file reproducibility pipeline for Project 2A.

Real data:
    python project2a_conjunction_risk_full_analysis.py --data-dir PATH_TO_ESA_FILES

Demo smoke test:
    python project2a_conjunction_risk_full_analysis.py --demo --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

SEED = 42
FLOOR = -30.0
HORIZONS = [1.0, 2.0, 3.0, 5.0, 6.0]
SNAPSHOT = [
    "time_to_tca",
    "risk",
    "max_risk_estimate",
    "max_risk_scaling",
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",
    "c_object_type",
    "mahalanobis_distance",
    "c_position_covariance_det",
    "c_sigma_r",
    "c_sigma_t",
    "c_sigma_n",
    "c_obs_used",
    "SSN",
]
TREND_SOURCES = [
    "risk",
    "max_risk_estimate",
    "miss_distance",
    "mahalanobis_distance",
    "c_position_covariance_det",
]


def make_demo(folder: Path) -> None:
    """Create synthetic training/test event histories for a smoke test only."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8)

    def events(count: int, start: int) -> pd.DataFrame:
        rows = []
        for event_id in range(start, start + count):
            final = FLOOR if rng.random() < 0.65 else rng.uniform(-12, -3)
            message_count = rng.integers(5, 10)
            times = np.sort(rng.uniform(0.05, 6.8, message_count))[::-1]
            object_type = rng.choice(["PAYLOAD", "DEBRIS", "ROCKET BODY"])
            for time_to_tca in times:
                risk = np.clip(final - rng.normal(0.3 * time_to_tca, 1.0), FLOOR, -1)
                rows.append(
                    [
                        event_id,
                        time_to_tca,
                        risk,
                        risk + rng.normal(0, 0.4),
                        rng.uniform(0.5, 2),
                        rng.uniform(50, 20000),
                        rng.uniform(100, 15000),
                        rng.normal(0, 1000),
                        rng.normal(0, 1000),
                        rng.normal(0, 1000),
                        rng.normal(0, 5),
                        rng.normal(0, 5),
                        rng.normal(0, 5),
                        object_type,
                        rng.uniform(0.1, 10),
                        10 ** rng.uniform(1, 8),
                        rng.uniform(0.1, 100),
                        rng.uniform(0.1, 100),
                        rng.uniform(0.1, 100),
                        rng.integers(5, 100),
                        rng.uniform(20, 200),
                    ]
                )
        return pd.DataFrame(rows, columns=["event_id"] + SNAPSHOT)

    train = events(600, 0)
    test = events(150, 10000)
    final_indices = test.groupby("event_id").time_to_tca.idxmin()
    labels = test.loc[final_indices, ["event_id", "risk"]].rename(
        columns={"risk": "true_risk"}
    )
    train.to_csv(folder / "train_data.csv", index=False)
    test.to_csv(folder / "test_data.csv", index=False)
    labels.to_csv(folder / "test_data_private.csv", index=False)


def load_data(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three expected ESA-format input files."""
    folder = Path(folder)
    train = pd.read_csv(folder / "train_data.csv")
    test = pd.read_csv(folder / "test_data.csv")
    labels = pd.read_csv(folder / "test_data_private.csv")

    for name, dataframe in [("train", train), ("test", test)]:
        missing = {"event_id", "time_to_tca", "risk"} - set(dataframe.columns)
        if missing:
            raise ValueError(f"{name} missing {missing}")
    missing_labels = {"event_id", "true_risk"} - set(labels.columns)
    if missing_labels:
        raise ValueError(f"labels missing {missing_labels}")
    return train, test, labels


def history(raw: pd.DataFrame, horizon: float) -> pd.DataFrame:
    """Calculate sequence summaries using only messages available at a horizon."""
    available = raw[raw.time_to_tca >= horizon].sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    )
    grouped = available.groupby("event_id")
    output = pd.DataFrame(
        {
            "available_cdm_count": grouped.size(),
            "history_span_days": grouped.time_to_tca.max()
            - grouped.time_to_tca.min(),
        }
    )
    for column in TREND_SOURCES:
        if column not in available:
            continue
        first = grouped[column].first()
        last = grouped[column].last()
        output[f"{column}_history_std"] = grouped[column].std()
        output[f"{column}_last_minus_first"] = last - first
        output[f"{column}_rate_per_day"] = (last - first) / output[
            "history_span_days"
        ].replace(0, np.nan)
    return output.reset_index()


def build(
    raw: pd.DataFrame, horizon: float, labels: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Construct one leakage-aware modeling row per eligible event."""
    eligible = raw[raw.time_to_tca >= horizon]
    if eligible.empty:
        return pd.DataFrame()

    snapshot_indices = eligible.groupby("event_id").time_to_tca.idxmin()
    snapshot = raw.loc[snapshot_indices].copy()
    output = snapshot.merge(
        history(raw, horizon), on="event_id", validate="one_to_one"
    )

    if labels is None:
        final_indices = raw.groupby("event_id").time_to_tca.idxmin()
        target = raw.loc[final_indices, ["event_id", "risk"]].rename(
            columns={"risk": "target_final_risk"}
        )
    else:
        target = (
            labels[["event_id", "true_risk"]]
            .drop_duplicates("event_id")
            .rename(columns={"true_risk": "target_final_risk"})
        )

    output = output.merge(target, on="event_id", validate="one_to_one")
    assert output.event_id.is_unique
    assert (output.time_to_tca >= horizon).all()
    return output


def signed_log(values: np.ndarray) -> np.ndarray:
    """Apply a signed log1p transform while protecting against overflow."""
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -1e300, 1e300)
    return np.sign(values) * np.log1p(np.abs(values))


def preprocessor(features: list[str]) -> ColumnTransformer:
    """Build numerical and categorical preprocessing."""
    categorical = [column for column in features if column == "c_object_type"]
    numeric = [column for column in features if column not in categorical]
    parts = []
    if numeric:
        parts.append(
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        (
                            "log",
                            FunctionTransformer(
                                signed_log, feature_names_out="one-to-one"
                            ),
                        ),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(parts)


def regression_model(features: list[str], quick: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("pre", preprocessor(features)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=30 if quick else 60,
                    min_samples_leaf=4,
                    max_features=0.8,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def classification_model(features: list[str], quick: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("pre", preprocessor(features)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=30 if quick else 60,
                    min_samples_leaf=3,
                    max_features=0.8,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** 0.5,
        "R2": r2_score(y_true, prediction),
    }


def classification_metrics(
    y_true: pd.Series, probability: np.ndarray
) -> dict[str, float]:
    predicted_class = (probability >= 0.5).astype(int)
    output = {
        "accuracy": accuracy_score(y_true, predicted_class),
        "precision": precision_score(y_true, predicted_class, zero_division=0),
        "recall": recall_score(y_true, predicted_class, zero_division=0),
        "f1": f1_score(y_true, predicted_class, zero_division=0),
        "average_precision": average_precision_score(y_true, probability),
    }
    output["roc_auc"] = (
        roc_auc_score(y_true, probability) if pd.Series(y_true).nunique() > 1 else np.nan
    )
    return output


def fit_two_stage(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    quick: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the hurdle-style classifier/regressor and return predictions."""
    nonfloor = (training.target_final_risk > FLOOR).astype(int)
    if nonfloor.nunique() < 2 or nonfloor.sum() == 0:
        raise ValueError("Two-stage modeling requires floor and non-floor events")
    classifier = classification_model(features, quick)
    regressor = regression_model(features, quick)
    classifier.fit(training[features], nonfloor)
    regressor.fit(
        training.loc[nonfloor == 1, features],
        training.loc[nonfloor == 1, "target_final_risk"],
    )
    probability = classifier.predict_proba(evaluation[features])[:, 1]
    conditional = regressor.predict(evaluation[features])
    prediction = FLOOR * (1 - probability) + conditional * probability
    return prediction, probability


def run(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: pd.DataFrame,
    output: Path,
    quick: bool = False,
) -> None:
    """Run fixed-horizon single-stage and two-stage experiments."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    rows = []
    classification_rows = []
    availability_rows = []
    primary_predictions = None

    for horizon in HORIZONS:
        training = build(train, horizon)
        official = build(test, horizon, labels) if horizon >= 2 else pd.DataFrame()
        availability_rows.append(
            {
                "horizon": horizon,
                "training_events": len(training),
                "official_events": len(official),
                "training_floor_fraction": (
                    (training.target_final_risk == FLOOR).mean()
                    if len(training)
                    else np.nan
                ),
                "official_floor_fraction": (
                    (official.target_final_risk == FLOOR).mean()
                    if len(official)
                    else np.nan
                ),
            }
        )

        trend_features = SNAPSHOT + [
            column
            for column in training.columns
            if column.endswith(
                ("_history_std", "_last_minus_first", "_rate_per_day")
            )
        ] + ["available_cdm_count", "history_span_days"]
        snapshot_features = [column for column in SNAPSHOT if column in training]
        trend_features = list(
            dict.fromkeys(column for column in trend_features if column in training)
        )

        train_index, validation_index = train_test_split(
            np.arange(len(training)), test_size=0.2, random_state=SEED
        )
        internal_train = training.iloc[train_index]
        validation = training.iloc[validation_index]

        for label, features in [
            ("snapshot", snapshot_features),
            ("snapshot_plus_trends", trend_features),
        ]:
            model = regression_model(features, quick)
            model.fit(internal_train[features], internal_train.target_final_risk)
            prediction = model.predict(validation[features])
            rows.append(
                {
                    "horizon": horizon,
                    "split": "internal_validation",
                    "feature_set": label,
                    "model": "single_stage",
                    **regression_metrics(validation.target_final_risk, prediction),
                }
            )

            if len(official):
                model.fit(training[features], training.target_final_risk)
                official_prediction = model.predict(official[features])
                rows.append(
                    {
                        "horizon": horizon,
                        "split": "official_test",
                        "feature_set": label,
                        "model": "single_stage",
                        **regression_metrics(
                            official.target_final_risk, official_prediction
                        ),
                    }
                )
                if horizon == 2 and label == "snapshot_plus_trends":
                    primary_predictions = pd.DataFrame(
                        {
                            "event_id": official.event_id,
                            "actual": official.target_final_risk,
                            "single_stage_prediction": official_prediction,
                        }
                    )

        two_stage_prediction, probability = fit_two_stage(
            internal_train, validation, trend_features, quick
        )
        rows.append(
            {
                "horizon": horizon,
                "split": "internal_validation",
                "feature_set": "snapshot_plus_trends",
                "model": "two_stage",
                **regression_metrics(
                    validation.target_final_risk, two_stage_prediction
                ),
            }
        )
        classification_rows.append(
            {
                "horizon": horizon,
                "split": "internal_validation",
                **classification_metrics(
                    (validation.target_final_risk > FLOOR).astype(int), probability
                ),
            }
        )

        if len(official):
            two_stage_prediction, probability = fit_two_stage(
                training, official, trend_features, quick
            )
            rows.append(
                {
                    "horizon": horizon,
                    "split": "official_test",
                    "feature_set": "snapshot_plus_trends",
                    "model": "two_stage",
                    **regression_metrics(
                        official.target_final_risk, two_stage_prediction
                    ),
                }
            )
            classification_rows.append(
                {
                    "horizon": horizon,
                    "split": "official_test",
                    **classification_metrics(
                        (official.target_final_risk > FLOOR).astype(int), probability
                    ),
                }
            )
            if horizon == 2 and primary_predictions is not None:
                primary_predictions["two_stage_prediction"] = two_stage_prediction
                primary_predictions["probability_nonfloor"] = probability

    results = pd.DataFrame(rows)
    results.to_csv(output / "horizon_results.csv", index=False)
    pd.DataFrame(classification_rows).to_csv(
        output / "classification_results.csv", index=False
    )
    pd.DataFrame(availability_rows).to_csv(output / "availability.csv", index=False)
    if primary_predictions is not None:
        primary_predictions.to_csv(
            output / "official_test_predictions.csv", index=False
        )

    plot_data = results[
        (results.split == "official_test")
        & (results.feature_set == "snapshot_plus_trends")
    ]
    plt.figure(figsize=(8, 5))
    for name, group in plot_data.groupby("model"):
        plt.plot(group.horizon, group.MAE, "o-", label=name)
    plt.xlabel("Decision horizon before TCA (days)")
    plt.ylabel("MAE (log10 risk)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "figures/horizon_mae.png", dpi=250)
    plt.close()

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "training_rows": len(train),
        "training_events": train.event_id.nunique(),
        "test_rows": len(test),
        "test_events": test.event_id.nunique(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="project2a_outputs")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    folder = Path(arguments.data_dir)

    if arguments.demo:
        folder = Path("project2a_demo_data")
        make_demo(folder)
        print(f"Synthetic demo data created: {folder}")

    train, test, labels = load_data(folder)
    run(train, test, labels, Path(arguments.output), arguments.quick)
    print(f"Outputs saved to {arguments.output}")


if __name__ == "__main__":
    main()
