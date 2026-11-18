"""Document actual fixed cleaning and feature roles without fitting transformations."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import BINARY, NOMINAL, engineer_features, feature_profiles, load_data


if __name__ == "__main__":
    raw = pd.read_csv(ROOT / "data/raw/students_dropout.csv", sep=";")
    cleaned = load_data()
    engineered = engineer_features(cleaned)
    profiles = feature_profiles(engineered)
    metadata = json.loads((ROOT / "data/source_metadata.json").read_text())["data"]
    definitions = {item["name"].strip().casefold(): item.get("description") or "" for item in metadata["variables"]}
    dictionary = []
    for column in engineered:
        role = ("target" if column == "Target" else "nominal" if column in NOMINAL else
                "binary" if column in BINARY or column.endswith("(zero_enrollment)") else
                "ordinal" if column == "Application order" else "quantitative")
        group = "semester_1" if "1st sem" in column else "semester_2" if "2nd sem" in column else "non_semester"
        description = definitions.get(column.casefold(), "")
        if column not in cleaned:
            description = ("Indicator: enrolled count equals zero." if column.endswith("(zero_enrollment)") else
                           "Row-wise ratio to enrolled count; zero if denominator is zero, with separate indicator.")
        dictionary.append({"feature": column, "role": role, "feature_group": group,
                           "kind": "source" if column in cleaned else "engineered", "definition": description,
                           "included_main": column in profiles["plus_semester_1"]})
    records = [
        {"check": "source shape", "observed": f"{len(raw)} rows, {raw.shape[1]} columns", "action": "preserve every source row and target"},
        {"check": "empty cells", "observed": int(raw.isna().sum().sum()), "action": "none present; no source imputation"},
        {"check": "full duplicates", "observed": int(raw.duplicated().sum()), "action": "none present; no rows removed"},
        {"check": "header whitespace", "observed": sum(column != column.strip() for column in raw.columns), "action": "strip header whitespace in modelling memory; source bytes preserved"},
        {"check": "nominal codes", "observed": len(NOMINAL), "action": "cast to categorical strings; train-fitted one-hot encoding"},
        {"check": "binary codes", "observed": len(BINARY), "action": "validate each observed value is 0 or 1"},
        {"check": "financial timestamps", "observed": "Debtor; Tuition fees up to date", "action": "exclude from every core feature profile"},
        {"check": "target integrity", "observed": str(cleaned.Target.value_counts().to_dict()), "action": "preserve all three outcome classes; no imputation or binary collapse"},
        {"check": "feature engineering", "observed": engineered.shape[1] - cleaned.shape[1], "action": "fixed row-wise ratios and zero-enrollment flags; no fitted statistics"},
        {"check": "outliers and zero grades", "observed": "valid source extremes retained", "action": "no target-driven trimming or automatic deletion of zeros"},
    ]
    for column in ("Mother's qualification", "Father's qualification"):
        records.append({"check": f"{column}: unknown code 34", "observed": int((raw[column] == 34).sum()), "action": "map to explicit Unknown category"})
    for column in ("Mother's occupation", "Father's occupation"):
        records.append({"check": f"{column}: blank code 99", "observed": int((raw[column] == 99).sum()), "action": "map to explicit Unknown category"})
    destination = ROOT / "artifacts/tables"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dictionary).to_csv(destination / "feature_dictionary.csv", index=False)
    pd.DataFrame(records).to_csv(destination / "cleaning_log.csv", index=False)
    print(f"Saved feature dictionary ({len(dictionary)} columns including target) and cleaning log.")
