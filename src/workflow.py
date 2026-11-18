"""Predeclared training, validation selection, one frozen test comparison and analysis."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .data import ROOT, engineer_features, feature_profiles, fixed_split, load_data, preprocessor, read_config, write_json
from .plots import eda, result_figures


ARTIFACTS = ROOT / "artifacts"
TABLES = ARTIFACTS / "tables"
MODELS = ARTIFACTS / "models"
CLASSES = ["Dropout", "Enrolled", "Graduate"]
GRIDS = {
    "Logistic Regression": {"model__C": [.1, 1., 10.], "model__class_weight": [None, "balanced"]},
    "Decision Tree": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [5, 15],
                      "model__class_weight": [None, "balanced"]},
    "KNN": {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]},
}
ANN_CANDIDATES = [
    {"name": "ann_32", "hidden": [32], "alpha": .001, "balanced": False},
    {"name": "ann_64_32", "hidden": [64, 32], "alpha": .001, "balanced": False},
    {"name": "ann_32_weighted", "hidden": [32], "alpha": .01, "balanced": True},
    {"name": "ann_64_32_weighted", "hidden": [64, 32], "alpha": .01, "balanced": True},
]
CLUSTER_FEATURES = [f"Curricular units 1st sem ({name})" for name in
                    ("enrolled", "evaluations", "approved", "grade", "without evaluations")]


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y: pd.Series | np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(y, prediction, labels=CLASSES, zero_division=0)
    return {"macro_f1": float(f1_score(y, prediction, labels=CLASSES, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "dropout_precision": float(precision[0]), "dropout_recall": float(recall[0])}


def plan() -> dict:
    config = read_config()
    value = {
        "dataset_sha256": config["dataset"]["sha256"], "seed": 42,
        "feature_ablation_plan": config["feature_ablation_plan"],
        "split": "stratified 70/15/15 with seed 42 at both split stages",
        "cv": {"folds": 5, "shuffle": True, "seed": 42, "score": "f1_macro", "scope": "training only"},
        "classical_grids": GRIDS, "ann_candidates": ANN_CANDIDATES,
        "ann_training": {"learning_rate": .001, "batch_size": 64, "maximum_epochs": 200,
                         "patience": 25, "min_macro_f1_improvement": .0001,
                         "epoch_selection": "highest validation Macro F1; exact ties prefer lower validation log loss",
                         "candidate_selection": "highest validation Macro F1; exact ties retain first declared candidate",
                         "activation": "ReLU hidden layers, softmax multiclass output",
                         "optimizer": "Adam, beta_1=.9, beta_2=.999, epsilon=1e-8",
                         "loss": "cross-entropy with L2; weighted training for declared weighted candidates",
                         "displayed_curves": "unweighted train/validation cross-entropy and Macro F1"},
        "clustering": {"features": CLUSTER_FEATURES, "k": [2, 3, 4, 5, 6], "n_init": 20,
                       "seeds": [42, 43, 44], "silhouette_sample": 1200,
                       "selection": "maximum seed-42 silhouette among k with minimum cluster share >=5%; if none, maximum silhouette with small-cluster limitation disclosed",
                       "tie_break": "smaller k", "target_used_for_selection": False},
        "pca": {"fit_scope": "training", "features": CLUSTER_FEATURES, "scaling": "StandardScaler",
                "solver": "full", "retention_threshold": .95},
        "main_model_selection": "highest validation Macro F1; exact ties retain declaration order",
        "ablation": "Same split, same logistic grid and train-only CV for each fixed feature profile",
        "test_policy": "No selection or post-test retuning; all main models fit train only, without train+validation refit",
        "core_code_sha256": {name: sha(ROOT / "src" / name) for name in ("data.py", "workflow.py")},
    }
    encoded = json.dumps(value, sort_keys=True).encode()
    return {**value, "plan_sha256": hashlib.sha256(encoded).hexdigest()}


def classical(name: str, columns: list[str]) -> Pipeline:
    estimators = {
        "Dummy": DummyClassifier(strategy="most_frequent", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
    }
    return Pipeline([("preprocess", preprocessor(columns, scale=name != "Decision Tree")), ("model", estimators[name])])


def tune(name: str, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, tag: str) -> tuple[Pipeline, dict]:
    started = time.monotonic()
    search = GridSearchCV(classical(name, list(X.columns)), GRIDS[name], scoring="f1_macro", cv=cv,
                          refit=True, n_jobs=1, return_train_score=True, error_score="raise")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(TABLES / f"cv_{tag}.csv", index=False)
    selected = results.iloc[search.best_index_]
    metadata = {"parameters": search.best_params_, "cv_macro_f1": float(search.best_score_),
                "cv_std": float(selected["std_test_score"]), "candidates": len(results),
                "fit_seconds": round(time.monotonic() - started, 3),
                "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught)}
    print(f"  {tag}: CV Macro F1={search.best_score_:.4f}, {len(results)} candidates", flush=True)
    return search.best_estimator_, metadata


def train_ann(X: pd.DataFrame, y: pd.Series, validation_X: pd.DataFrame, validation_y: pd.Series) -> tuple[Pipeline, dict]:
    transform = preprocessor(list(X.columns), scale=True)
    train_array = transform.fit_transform(X)
    validation_array = transform.transform(validation_X)
    rows, candidates = [], []
    selected_network = None
    selected_metadata = None
    for candidate in ANN_CANDIDATES:
        started = time.monotonic()
        network = MLPClassifier(hidden_layer_sizes=tuple(candidate["hidden"]), activation="relu", solver="adam",
                                alpha=candidate["alpha"], batch_size=64, learning_rate_init=.001,
                                random_state=42, shuffle=True, early_stopping=False)
        weights = compute_sample_weight("balanced", y) if candidate["balanced"] else None
        best_score, best_loss, best_epoch, stale = -1., float("inf"), 0, 0
        best_network = None
        for epoch in range(1, 201):
            network.partial_fit(train_array, y, classes=np.array(CLASSES), sample_weight=weights)
            train_prob = network.predict_proba(train_array)
            validation_prob = network.predict_proba(validation_array)
            train_f1 = f1_score(y, network.classes_[np.argmax(train_prob, axis=1)], average="macro", zero_division=0)
            validation_f1 = f1_score(validation_y, network.classes_[np.argmax(validation_prob, axis=1)], average="macro", zero_division=0)
            validation_loss = log_loss(validation_y, validation_prob, labels=CLASSES)
            rows.append({"candidate": candidate["name"], "epoch": epoch,
                         "train_log_loss": float(log_loss(y, train_prob, labels=CLASSES)),
                         "validation_log_loss": float(validation_loss), "train_macro_f1": float(train_f1),
                         "validation_macro_f1": float(validation_f1)})
            substantial = validation_f1 > best_score + .0001
            improved = validation_f1 > best_score or (validation_f1 == best_score and validation_loss < best_loss)
            if improved:
                best_score, best_loss, best_epoch = float(validation_f1), float(validation_loss), epoch
                best_network = copy.deepcopy(network)
            stale = 0 if substantial else stale + 1
            if stale >= 25:
                break
        record = {**candidate, "best_epoch": best_epoch, "stopped_epoch": epoch,
                  "validation_macro_f1": best_score, "validation_log_loss": best_loss,
                  "fit_seconds": round(time.monotonic() - started, 3)}
        candidates.append(record)
        if selected_metadata is None or best_score > selected_metadata["validation_macro_f1"]:
            selected_network, selected_metadata = best_network, record
        print(f"  {candidate['name']}: validation Macro F1={best_score:.4f}, restored epoch {best_epoch}/{epoch}", flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "ann_history.csv", index=False)
    pd.DataFrame(candidates).to_csv(TABLES / "ann_candidates.csv", index=False)
    fitted = Pipeline([("preprocess", transform), ("model", selected_network)])
    parameters = sum(array.size for array in selected_network.coefs_) + sum(array.size for array in selected_network.intercepts_)
    metadata = {**selected_metadata, "selected_candidate": selected_metadata["name"],
                "input_dimension": int(train_array.shape[1]), "parameter_count": int(parameters),
                "output_classes": selected_network.classes_.tolist(), "training_rows": len(y)}
    return fitted, metadata


def unsupervised(training: pd.DataFrame) -> dict:
    transform = StandardScaler()
    array = transform.fit_transform(training[CLUSTER_FEATURES])
    options, fitted = [], {}
    for k in range(2, 7):
        models = [KMeans(n_clusters=k, n_init=20, random_state=seed).fit(array) for seed in (42, 43, 44)]
        main = models[0]
        labels = main.labels_
        counts = np.bincount(labels, minlength=k)
        stability = [adjusted_rand_score(a.labels_, b.labels_) for a, b in itertools.combinations(models, 2)]
        row = {"k": k, "inertia": float(main.inertia_),
               "silhouette": float(silhouette_score(array, labels, sample_size=min(1200, len(array)), random_state=42)),
               "minimum_cluster_fraction": float(counts.min() / len(array)), "mean_seed_ari": float(np.mean(stability))}
        options.append(row)
        fitted[k] = main
    eligible = [item for item in options if item["minimum_cluster_fraction"] >= .05]
    selection = max(eligible or options, key=lambda item: (item["silhouette"], -item["k"]))
    model = fitted[selection["k"]]
    labels = model.labels_
    pd.DataFrame(options).to_csv(TABLES / "cluster_selection.csv", index=False)
    pca = PCA(svd_solver="full").fit(array)
    projected = pca.transform(array)
    cumulative = pca.explained_variance_ratio_.cumsum()
    retained = int(np.searchsorted(cumulative, .95) + 1)
    pd.DataFrame({"component": np.arange(1, len(cumulative) + 1),
                  "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_variance": cumulative}).to_csv(TABLES / "pca_variance.csv", index=False)
    pd.DataFrame(pca.components_.T, index=CLUSTER_FEATURES,
                 columns=[f"PC{i}" for i in range(1, len(cumulative) + 1)]).to_csv(TABLES / "pca_loadings.csv")
    pd.DataFrame({"row_id": training.index, "cluster": labels, "PC1": projected[:, 0], "PC2": projected[:, 1],
                  "Target": training.Target.to_numpy()}).to_csv(TABLES / "training_pca_projection.csv", index=False)
    profiles = []
    for label in range(selection["k"]):
        subset = training.iloc[np.flatnonzero(labels == label)]
        record = {"cluster": label, "n": len(subset)}
        record.update({f"mean_{column}": float(subset[column].mean()) for column in CLUSTER_FEATURES})
        record.update({f"share_{name}": float((subset.Target == name).mean()) for name in CLASSES})
        profiles.append(record)
    pd.DataFrame(profiles).to_csv(TABLES / "cluster_profiles.csv", index=False)
    joblib.dump({"features": CLUSTER_FEATURES, "scaler": transform, "kmeans": model, "pca": pca}, MODELS / "unsupervised.joblib", compress=3)
    return {**selection, "features": CLUSTER_FEATURES, "training_rows": len(training),
            "size_constraint_satisfied": bool(eligible), "pc1_pc2_variance": float(cumulative[1]),
            "components_for_95_percent": retained, "cluster_profiles": profiles,
            "interpretation_limit": "Seed stability and silhouette do not prove natural classes; PCA components are not causes."}


def analyse_errors(model: Pipeline, frame: pd.DataFrame, y: pd.Series, columns: list[str]) -> dict:
    prediction = model.predict(frame[columns])
    probability = model.predict_proba(frame[columns])
    results = pd.DataFrame({"row_id": frame.index, "actual": y.to_numpy(), "predicted": prediction,
                            "confidence": probability.max(axis=1), "incorrect": y.to_numpy() != prediction})
    results.to_csv(TABLES / "selected_test_predictions.csv", index=False)
    results.loc[results.incorrect].sort_values("confidence", ascending=False).to_csv(TABLES / "selected_test_errors.csv", index=False)
    matrix = confusion_matrix(y, prediction, labels=CLASSES)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(TABLES / "selected_confusion_matrix.csv")
    groups = {
        "age": pd.cut(frame["Age at enrollment"], [-np.inf, 20, 25, np.inf], labels=["<=20", "21-25", ">25"]).astype(str),
        "gender_code": frame["Gender"].astype(str),
        "scholarship": frame["Scholarship holder"].astype(str),
        "zero_passed_sem1": (frame["Curricular units 1st sem (approved)"] == 0).astype(str),
    }
    records = []
    for group, values in groups.items():
        for value in sorted(values.unique()):
            mask = values.to_numpy() == value
            records.append({"group": group, "value": value, "n": int(mask.sum()),
                            "dropout_n": int((y.to_numpy()[mask] == "Dropout").sum()),
                            **score(y.to_numpy()[mask], prediction[mask])})
    pd.DataFrame(records).to_csv(TABLES / "test_error_groups.csv", index=False)
    return {"incorrect": int(results.incorrect.sum()), "test_rows": len(results),
            "dropout_predicted_graduate": int(matrix[0, 2]), "graduate_predicted_dropout": int(matrix[2, 0]),
            "group_analysis_limit": "Descriptive small-group results, not a formal fairness or causal analysis."}


def run_experiments(reuse: bool = True) -> dict:
    current_plan = plan()
    summary_path = ARTIFACTS / "results_summary.json"
    frozen_path = ARTIFACTS / "frozen_models.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen["plan_sha256"] != current_plan["plan_sha256"]:
            raise RuntimeError("A test comparison is already frozen under a different plan. Do not retune using the exposed test.")
    if reuse and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["plan_sha256"] != current_plan["plan_sha256"]:
            raise RuntimeError("Cached results differ from the current declared experiment.")
        for relative, digest in summary["artifact_sha256"].items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError(f"Frozen artifact changed: {relative}")
        print("Loaded verified frozen results; no models retrained and no new test selection.", flush=True)
        return summary

    started = time.monotonic()
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACTS / "experiment_plan.json", {"declared_at_utc": stamp(), **current_plan})
    frame = engineer_features(load_data())
    splits, manifest = fixed_split(frame)
    training, validation = frame.loc[splits["train"]], frame.loc[splits["validation"]]
    y_train, y_validation = training.Target, validation.Target
    profiles = feature_profiles(frame)
    main_columns = profiles[read_config()["feature_ablation_plan"]["main_profile"]]
    split_table = pd.crosstab(manifest.split, manifest.Target).reindex(index=["train", "validation", "test"], columns=CLASSES)
    split_table.to_csv(TABLES / "split_counts.csv")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_membership = []
    for fold, (_, indexes) in enumerate(cv.split(training[main_columns], y_train), start=1):
        fold_membership.extend({"row_id": int(training.index[i]), "validation_fold": fold} for i in indexes)
    pd.DataFrame(fold_membership).sort_values("row_id").to_csv(TABLES / "cv_fold_membership.csv", index=False)
    print(f"Split frozen: train={len(training)}, validation={len(validation)}, test={len(splits['test'])}", flush=True)

    with threadpool_limits(limits=2):
        eda_summary = eda(training)
        print("Training-only EDA complete: five figures.", flush=True)
        estimators, model_info, validation_scores = {}, {}, {}
        baseline = classical("Dummy", main_columns).fit(training[main_columns], y_train)
        estimators["Dummy"] = baseline
        model_info["Dummy"] = {"parameters": {"strategy": "most_frequent"}, "training_rows": len(training)}
        validation_scores["Dummy"] = score(y_validation, baseline.predict(validation[main_columns]))
        for name, tag in (("Logistic Regression", "logistic_main"), ("Decision Tree", "tree_main"), ("KNN", "knn_main")):
            fitted, info = tune(name, training[main_columns], y_train, cv, tag)
            estimators[name], model_info[name] = fitted, {**info, "training_rows": len(training)}
            validation_scores[name] = score(y_validation, fitted.predict(validation[main_columns]))
        print("Classical model search complete; fitting four ANN candidates.", flush=True)
        estimators["ANN"], model_info["ANN"] = train_ann(training[main_columns], y_train, validation[main_columns], y_validation)
        validation_scores["ANN"] = score(y_validation, estimators["ANN"].predict(validation[main_columns]))

        ablation_estimators, ablation_records = {}, []
        for profile, columns in profiles.items():
            if profile == "plus_semester_1":
                fitted, info = estimators["Logistic Regression"], model_info["Logistic Regression"]
            else:
                fitted, info = tune("Logistic Regression", training[columns], y_train, cv, f"logistic_{profile}")
            ablation_estimators[profile] = fitted
            ablation_records.append({"profile": profile, "input_features": len(columns), "cv_macro_f1": info["cv_macro_f1"],
                                     **{f"validation_{k}": v for k, v in score(y_validation, fitted.predict(validation[columns])).items()}})

        print("Feature-group comparison trained; fitting K-Means and PCA on train only.", flush=True)
        unsupervised_summary = unsupervised(training)
        selected = max(validation_scores, key=lambda name: validation_scores[name]["macro_f1"])
        importance = permutation_importance(estimators[selected], validation[main_columns], y_validation,
                                            scoring="f1_macro", n_repeats=5, random_state=42, n_jobs=1)
        pd.DataFrame({"feature": main_columns, "validation_importance_mean": importance.importances_mean,
                      "validation_importance_std": importance.importances_std}).sort_values(
                          "validation_importance_mean", ascending=False).to_csv(TABLES / "validation_permutation_importance.csv", index=False)

        model_paths = {}
        for name, fitted in estimators.items():
            relative = f"artifacts/models/{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(fitted, ROOT / relative, compress=3)
            model_paths[name] = relative
        for profile, fitted in ablation_estimators.items():
            joblib.dump(fitted, MODELS / f"ablation_{profile}.joblib", compress=3)
        frozen = {"frozen_at_utc": stamp(), "plan_sha256": current_plan["plan_sha256"],
                  "selected_model": selected, "selection_metric": "validation Macro F1",
                  "main_features": main_columns, "validation_scores": validation_scores,
                  "model_info": model_info, "test_used_for_selection": False,
                  "model_sha256": {name: sha(ROOT / path) for name, path in model_paths.items()}}
        # This record is written before obtaining any test predictions or metrics.
        write_json(frozen_path, frozen)
        print(f"Models frozen. Validation selected {selected}; now evaluating the held-out test.", flush=True)
        testing = frame.loc[splits["test"]]
        final_rows, class_rows, all_predictions = [], [], []
        for name, fitted in estimators.items():
            prediction = fitted.predict(testing[main_columns])
            test_metrics = score(testing.Target, prediction)
            final_rows.append({"model": name, "cv_macro_f1": model_info[name].get("cv_macro_f1"),
                               **{f"validation_{k}": v for k, v in validation_scores[name].items()},
                               **{f"test_{k}": v for k, v in test_metrics.items()},
                               "selected_on_validation": name == selected, "training_rows": len(training),
                               "encoded_features": int(fitted.named_steps["preprocess"].transform(training[main_columns].iloc[:1]).shape[1])})
            p, r, f, support = precision_recall_fscore_support(testing.Target, prediction, labels=CLASSES, zero_division=0)
            class_rows.extend({"model": name, "class": label, "precision": float(p[i]), "recall": float(r[i]),
                               "f1": float(f[i]), "support": int(support[i])} for i, label in enumerate(CLASSES))
            all_predictions.extend({"model": name, "row_id": int(index), "actual": actual, "predicted": predicted}
                                   for index, actual, predicted in zip(testing.index, testing.Target, prediction, strict=True))
        pd.DataFrame(final_rows).to_csv(TABLES / "final_comparison.csv", index=False)
        pd.DataFrame(class_rows).to_csv(TABLES / "test_class_metrics.csv", index=False)
        pd.DataFrame(all_predictions).to_csv(TABLES / "all_test_predictions.csv", index=False)
        for record in ablation_records:
            profile = record["profile"]
            prediction = ablation_estimators[profile].predict(testing[profiles[profile]])
            record.update({f"test_{k}": v for k, v in score(testing.Target, prediction).items()})
        pd.DataFrame(ablation_records).to_csv(TABLES / "feature_ablation.csv", index=False)
        error_summary = analyse_errors(estimators[selected], testing, testing.Target, main_columns)

    summary = {"status": "numerical_experiments_complete_team_review_pending", "completed_at_utc": stamp(),
               "plan_sha256": current_plan["plan_sha256"], "dataset_sha256": current_plan["dataset_sha256"],
               "split_counts": {name: len(indexes) for name, indexes in splits.items()},
               "main_profile": "plus_semester_1", "raw_predictors_main": 28, "engineered_predictors_main": len(main_columns),
               "selected_model": selected, "main_model_info": model_info, "ann": model_info["ANN"],
               "eda": eda_summary, "unsupervised": unsupervised_summary,
               "final_comparison": final_rows, "feature_ablation": ablation_records, "error_analysis": error_summary,
               "elapsed_seconds": round(time.monotonic() - started, 2),
               "protocol": "Shared stratified split; all supervised models fit train only; no post-test selection or refit.",
               "limitations": ["One institution and historical cohorts; no independent prospective validation.",
                               "No individual dropout dates; semester profiles are retrospective information comparisons.",
                               "Financial status timestamps unresolved; Debtor and Tuition fees up to date excluded.",
                               "Enrolled is a distinct end-of-normal-duration status, not guaranteed future failure.",
                               "Feature importance and group differences are associations, not causes.",
                               "Student technical contributions remain to be performed and recorded by the team."],
               "artifact_sha256": {}}
    result_figures(summary)
    protected = list(TABLES.glob("*.csv")) + list(MODELS.glob("*.joblib")) + [ROOT / "data/processed/split_manifest.csv", frozen_path]
    summary["artifact_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in sorted(protected)}
    write_json(summary_path, summary)
    print(pd.DataFrame(final_rows)[["model", "validation_macro_f1", "test_macro_f1", "test_accuracy"]].to_string(index=False), flush=True)
    print(f"Complete in {summary['elapsed_seconds']:.1f}s. Artifacts: {ARTIFACTS}", flush=True)
    return summary
