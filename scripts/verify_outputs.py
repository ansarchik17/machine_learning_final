"""Verify numerical artifacts, serialized models and leakage-sensitive contracts."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NOMINAL, engineer_features, feature_profiles, fixed_split, load_data, write_json
from src.workflow import CLASSES, plan, score, sha


def main() -> None:
    summary = json.loads((ROOT / "artifacts/results_summary.json").read_text())
    frozen = json.loads((ROOT / "artifacts/frozen_models.json").read_text())
    assert summary["plan_sha256"] == frozen["plan_sha256"] == plan()["plan_sha256"]
    for relative, digest in summary["artifact_sha256"].items():
        assert sha(ROOT / relative) == digest, relative
    assert frozen["test_used_for_selection"] is False
    assert dt.datetime.fromisoformat(frozen["frozen_at_utc"]) <= dt.datetime.fromisoformat(summary["completed_at_utc"])
    frame = engineer_features(load_data())
    splits, _ = fixed_split(frame)
    profiles = feature_profiles(frame)
    columns = profiles["plus_semester_1"]
    train = frame.loc[splits["train"]]
    validation = frame.loc[splits["validation"]]
    testing = frame.loc[splits["test"]]
    predictions = pd.read_csv(ROOT / "artifacts/tables/all_test_predictions.csv")
    table = pd.read_csv(ROOT / "artifacts/tables/final_comparison.csv")
    folds = pd.read_csv(ROOT / "artifacts/tables/cv_fold_membership.csv")
    assert set(folds.row_id) == set(train.index) and not folds.row_id.duplicated().any()
    assert set(folds.validation_fold) == {1, 2, 3, 4, 5}
    assert not predictions.duplicated(["model", "row_id"]).any()
    for name in table.model:
        subset = predictions.loc[predictions.model == name].set_index("row_id").loc[testing.index]
        assert set(subset.index) == set(testing.index)
        np.testing.assert_array_equal(subset.actual, testing.Target)
        model = joblib.load(ROOT / f"artifacts/models/{name.lower().replace(' ', '_')}.joblib")
        np.testing.assert_array_equal(model.predict(testing[columns]), subset.predicted)
        expected = table.loc[table.model == name].iloc[0]
        for metric, value in score(testing.Target, subset.predicted.to_numpy()).items():
            np.testing.assert_allclose(value, expected[f"test_{metric}"], atol=1e-12)
        validation_f1 = f1_score(validation.Target, model.predict(validation[columns]), average="macro")
        np.testing.assert_allclose(validation_f1, expected.validation_macro_f1, atol=1e-12)
        transform = model.named_steps["preprocess"]
        numeric = transform.named_transformers_["numeric"]
        selected = next(cols for key, _, cols in transform.transformers_ if key == "numeric")
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, train[selected].median(), atol=1e-12)
        if "scaler" in numeric.named_steps:
            assert numeric.named_steps["scaler"].n_samples_seen_ == len(train)
            np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, train[selected].mean(), atol=1e-12)
        for column, categories in zip([c for c in columns if c in NOMINAL], transform.named_transformers_["nominal"].categories_, strict=True):
            assert set(categories) == set(train[column])
    selected_on_validation = table.loc[table.validation_macro_f1.idxmax(), "model"]
    assert summary["selected_model"] == frozen["selected_model"] == selected_on_validation
    class_metrics = pd.read_csv(ROOT / "artifacts/tables/test_class_metrics.csv")
    for name in table.model:
        supports = class_metrics.loc[class_metrics.model == name].set_index("class").support
        assert supports.to_dict() == testing.Target.value_counts().to_dict()
    confusion = pd.read_csv(ROOT / "artifacts/tables/selected_confusion_matrix.csv", index_col=0)
    assert confusion.to_numpy().sum() == len(testing)
    history = pd.read_csv(ROOT / "artifacts/tables/ann_history.csv")
    chosen = history.loc[history.candidate == summary["ann"]["selected_candidate"]]
    best = chosen.sort_values(["validation_macro_f1", "validation_log_loss"], ascending=[False, True]).iloc[0]
    assert int(best.epoch) == summary["ann"]["best_epoch"]
    np.testing.assert_allclose(best.validation_macro_f1, table.loc[table.model == "ANN", "validation_macro_f1"].iloc[0], atol=1e-12)
    unsupervised = joblib.load(ROOT / "artifacts/models/unsupervised.joblib")
    assert "Target" not in unsupervised["features"]
    assert unsupervised["scaler"].n_samples_seen_ == unsupervised["pca"].n_samples_ == len(train)
    np.testing.assert_allclose(unsupervised["scaler"].mean_, train[unsupervised["features"]].mean(), atol=1e-12)
    assert len(unsupervised["kmeans"].labels_) == len(train)
    variance = pd.read_csv(ROOT / "artifacts/tables/pca_variance.csv")
    retained = summary["unsupervised"]["components_for_95_percent"]
    assert variance.cumulative_variance.iloc[retained - 1] >= .95
    if retained > 1:
        assert variance.cumulative_variance.iloc[retained - 2] < .95
    result = {"verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "passed",
              "checks": ["source and artifact hashes", "disjoint split and CV membership", "test prediction identity",
                         "independently recomputed metrics", "model selection uses validation only",
                         "training-only preprocessing statistics and category vocabularies", "ANN checkpoint restoration",
                         "training-only clustering and PCA", "PCA variance retention"],
              "main_models_checked": len(table), "test_rows_per_model": len(testing),
              "scope": "These are implementation contracts, not proof of prospective validity or absence of all source-data biases."}
    write_json(ROOT / "artifacts/verification.json", result)
    print(f"Verification passed: {len(table)} models, {len(testing)} test rows each; saved metrics and training-only transformations agree.")


if __name__ == "__main__":
    main()
