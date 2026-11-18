"""Source validation, deterministic cleaning, feature groups and fixed splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
NOMINAL = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
BINARY = [
    "Daytime/evening attendance", "Displaced", "Educational special needs",
    "Gender", "Scholarship holder", "International",
]
SEMESTER_SUFFIXES = ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")


def read_config() -> dict:
    return json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_data() -> pd.DataFrame:
    """Preserve original bytes and apply only fixed, non-learned transformations."""
    config = read_config()
    source = ROOT / config["dataset"]["raw_path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["dataset"]["sha256"]:
        raise ValueError("Source hash differs from the pinned dataset.")
    frame = pd.read_csv(source, sep=";")
    frame.columns = frame.columns.str.strip()
    if frame.shape != (4424, 37) or frame.isna().any().any():
        raise ValueError("Unexpected source shape or missing cells.")
    if frame["Target"].value_counts().to_dict() != config["dataset"]["expected_class_counts"]:
        raise ValueError("Unexpected target values or counts.")
    frame.index.name = "row_id"
    # Codes retain their categorical meaning; unknown codes become explicit labels.
    for column in NOMINAL:
        frame[column] = frame[column].astype(str)
    for column in ("Mother's qualification", "Father's qualification"):
        frame[column] = frame[column].replace("34", "Unknown")
    for column in ("Mother's occupation", "Father's occupation"):
        frame[column] = frame[column].replace("99", "Unknown")
    for column in BINARY:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"Unexpected binary code in {column}")
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed row-wise academic rates, with explicit zero-enrollment indicators."""
    result = frame.copy()
    for semester in ("1st", "2nd"):
        prefix = f"Curricular units {semester} sem"
        enrolled = result[f"{prefix} (enrolled)"].to_numpy(dtype=float)
        for suffix, numerator in (
            ("pass_rate", "approved"),
            ("evaluations_per_unit", "evaluations"),
            ("without_evaluations_rate", "without evaluations"),
        ):
            result[f"{prefix} ({suffix})"] = np.divide(
                result[f"{prefix} ({numerator})"].to_numpy(dtype=float), enrolled,
                out=np.zeros(len(result)), where=enrolled != 0,
            )
        result[f"{prefix} (zero_enrollment)"] = (enrolled == 0).astype(int)
    return result


def feature_profiles(frame: pd.DataFrame) -> dict[str, list[str]]:
    config = read_config()["feature_ablation_plan"]
    return {
        profile["name"]: [
            column for column in frame.columns
            if column != "Target"
            and column not in config["exclude_from_all_core_profiles"]
            and not any(column.startswith(prefix) for prefix in profile["exclude_prefixes"])
        ] for profile in config["profiles"]
    }


def preprocessor(columns: list[str], scale: bool = True) -> ColumnTransformer:
    categories = [column for column in columns if column in NOMINAL]
    quantitative = [column for column in columns if column not in NOMINAL and column not in BINARY]
    binary = [column for column in columns if column in BINARY]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), quantitative),
        ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
        ("binary", "passthrough", binary),
    ], remainder="drop", sparse_threshold=0)


def fixed_split(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Create once, or verify an identical split; never silently reshuffle a test."""
    seed = read_config()["evaluation_plan"]["random_seed"]
    training, holdout = train_test_split(
        frame.index.to_numpy(), test_size=0.30, stratify=frame["Target"], random_state=seed,
    )
    validation, testing = train_test_split(
        holdout, test_size=0.50, stratify=frame.loc[holdout, "Target"], random_state=seed,
    )
    splits = {"train": np.sort(training), "validation": np.sort(validation), "test": np.sort(testing)}
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Split leakage: overlapping rows.")
    if set(np.concatenate(list(splits.values()))) != set(frame.index):
        raise AssertionError("Split does not cover the source data.")
    records = [{"row_id": int(i), "split": name, "Target": frame.loc[i, "Target"]}
               for name, indexes in splits.items() for i in indexes]
    manifest = pd.DataFrame(records).sort_values("row_id").reset_index(drop=True)
    destination = ROOT / "data/processed/split_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        pd.testing.assert_frame_equal(pd.read_csv(destination), manifest)
    else:
        manifest.to_csv(destination, index=False)
    return splits, manifest
