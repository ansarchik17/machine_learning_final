"""Download and audit the pinned UCI source data without fitting any models.

Uses the Python standard library only. The original CSV is preserved byte for
byte; normalized headers are used in memory for validation and reporting.
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "StudentOutcomeCourseProject/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))
    source = config["dataset"]
    raw_path = ROOT / source["raw_path"]

    if raw_path.exists():
        raw = raw_path.read_bytes()
    else:
        with zipfile.ZipFile(io.BytesIO(fetch(source["archive_url"]))) as archive:
            names = [name for name in archive.namelist() if Path(name).name == "data.csv"]
            if len(names) != 1:
                raise ValueError(f"Expected one data.csv in the UCI archive, found {names}")
            raw = archive.read(names[0])

    digest = hashlib.sha256(raw).hexdigest()
    if digest != source["sha256"]:
        raise ValueError("The CSV differs from the pinned source. Review the source version before proceeding.")

    records = list(csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"))
    original_headers, *values = records
    headers = [name.strip() for name in original_headers]
    if len(headers) != len(set(headers)):
        raise ValueError("Header normalization creates duplicate names.")
    if len(headers) != source["expected_predictors"] + 1 or headers.count(config["target"]) != 1:
        raise ValueError("Unexpected feature count or target column.")
    if len(values) != source["expected_rows"] or any(len(row) != len(headers) for row in values):
        raise ValueError("Unexpected number of rows or malformed CSV records.")

    rows = [dict(zip(headers, row, strict=True)) for row in values]
    target = config["target"]
    predictors = [column for column in headers if column != target]
    class_counts = dict(collections.Counter(row[target] for row in rows))
    if class_counts != source["expected_class_counts"]:
        raise ValueError(f"Unexpected class counts: {class_counts}")

    ranges = {}
    for column in predictors:
        numbers = [float(row[column]) for row in rows]
        ranges[column] = {"min": min(numbers), "max": max(numbers), "unique": len(set(numbers))}

    profiles = {}
    ablation = config["feature_ablation_plan"]
    for profile in ablation["profiles"]:
        selected = [
            column for column in predictors
            if column not in ablation["exclude_from_all_core_profiles"]
            and not any(column.startswith(prefix) for prefix in profile["exclude_prefixes"])
        ]
        profiles[profile["name"]] = {"predictor_count": len(selected), "predictors": selected}

    metadata_path = ROOT / source["metadata_path"]
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = json.loads(fetch(source["metadata_url"]))
    if metadata.get("status") != 200 or metadata.get("data", {}).get("uci_id") != source["uci_id"]:
        raise ValueError("Unexpected UCI metadata response.")

    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
    if not metadata_path.exists():
        save_json(metadata_path, metadata)

    report = {
        "status": "source_data_verified",
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_path": source["raw_path"],
        "source_archive_url": source["archive_url"],
        "sha256": digest,
        "row_count": len(rows),
        "predictor_count": len(predictors),
        "target_counts": class_counts,
        "empty_cell_count": sum(not value.strip() for row in values for value in row),
        "duplicate_full_rows": len(rows) - len({tuple(row) for row in values}),
        "duplicate_predictor_rows": len(rows) - len({tuple(row[column] for column in predictors) for row in rows}),
        "header_normalization": [
            {"original": before, "normalized": after}
            for before, after in zip(original_headers, headers, strict=True) if before != after
        ],
        "encoded_unknown_counts": {
            "Mother's qualification:34": sum(row["Mother's qualification"] == "34" for row in rows),
            "Father's qualification:34": sum(row["Father's qualification"] == "34" for row in rows),
            "Mother's occupation:99": sum(row["Mother's occupation"] == "99" for row in rows),
            "Father's occupation:99": sum(row["Father's occupation"] == "99" for row in rows)
        },
        "numeric_storage_ranges": ranges,
        "feature_profiles": profiles,
        "notes": [
            "Numeric storage does not imply quantitative meaning: categorical codes require appropriate encoding.",
            "No blank cells does not imply all information is known: coded unknown categories remain.",
            "Feature profiles are retrospective ablations, not proof of availability at an individual prediction date.",
            "This source audit does not fit transformers or models; the numerical study has a separate results summary."
        ]
    }
    report_path = ROOT / "artifacts" / "data_audit.json"
    save_json(report_path, report)
    print(f"Verified {len(rows):,} rows, {len(predictors)} predictors; classes: {class_counts}")
    print(f"Empty cells: {report['empty_cell_count']}; duplicate rows: {report['duplicate_full_rows']}")
    profile_summary = ", ".join("{}={}".format(name, item["predictor_count"]) for name, item in profiles.items())
    print(f"Feature profiles: {profile_summary}")
    print(f"Audit saved to {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
