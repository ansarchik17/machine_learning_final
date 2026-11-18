"""Readable training-only EDA and figures generated from frozen experiment tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT


CLASSES = ["Dropout", "Enrolled", "Graduate"]
COLORS = ["#c84b53", "#dfaa45", "#277f87"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(figure: plt.Figure, name: str) -> None:
    path = ROOT / "artifacts/figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def eda(training: pd.DataFrame) -> dict:
    counts = training["Target"].value_counts().reindex(CLASSES)
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(CLASSES, counts, color=COLORS)
    axis.bar_label(bars, labels=[f"{n:,}\n{n / len(training):.1%}" for n in counts], padding=5)
    axis.set(ylim=(0, counts.max() * 1.22), ylabel="Students", title="Training outcomes: all three classes matter")
    save(figure, "eda_01_class_balance")

    figure, axis = plt.subplots(figsize=(7, 4))
    boxes = axis.boxplot([training.loc[training.Target == c, "Age at enrollment"] for c in CLASSES],
                         tick_labels=CLASSES, patch_artist=True, showfliers=True)
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(.65)
    axis.set(ylabel="Age at enrollment", title="Age distributions overlap across outcomes (training only)")
    save(figure, "eda_02_age")

    figure, axis = plt.subplots(figsize=(7, 4))
    for label, color in zip(CLASSES, COLORS):
        axis.hist(training.loc[training.Target == label, "Curricular units 1st sem (approved)"],
                  bins=np.arange(-.5, 27.5), density=True, histtype="step", linewidth=2,
                  label=label, color=color)
    axis.set(xlabel="Passed curricular units, semester 1", ylabel="Within-class density",
             title="Early academic progress differs by eventual outcome")
    axis.legend()
    save(figure, "eda_03_passed_units")

    selected = ["Age at enrollment", "Admission grade", "Curricular units 1st sem (enrolled)",
                "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)", "Curricular units 1st sem (pass_rate)"]
    short = ["Age", "Admission", "Enrolled", "Evaluations", "Passed", "Mean grade", "Pass rate"]
    correlations = training[selected].corr()
    figure, axis = plt.subplots(figsize=(8, 6))
    plot = axis.imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(short)), short, rotation=45, ha="right")
    axis.set_yticks(range(len(short)), short)
    for i in range(len(short)):
        for j in range(len(short)):
            axis.text(j, i, f"{correlations.iloc[i, j]:.2f}", ha="center", va="center",
                      color="white" if abs(correlations.iloc[i, j]) > .65 else "#202020", fontsize=8)
    figure.colorbar(plot, ax=axis, label="Pearson correlation", shrink=.8)
    axis.set_title("Quantitative features share information (training only)")
    save(figure, "eda_04_correlations")

    course_counts = pd.crosstab(training["Course"], training["Target"]).reindex(columns=CLASSES, fill_value=0)
    course_counts = course_counts.loc[course_counts.sum(axis=1).sort_values(ascending=False).index]
    proportions = course_counts.div(course_counts.sum(axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(9, 7))
    left = np.zeros(len(proportions))
    for label, color in zip(CLASSES, COLORS):
        axis.barh(range(len(proportions)), proportions[label], left=left, color=color, label=label)
        left += proportions[label].to_numpy()
    axis.set_yticks(range(len(proportions)), [f"{code} (n={int(course_counts.loc[code].sum())})" for code in proportions.index])
    axis.invert_yaxis()
    axis.set(xlim=(0, 1), xlabel="Proportion within course", ylabel="Course code and sample size",
             title="Outcomes vary by programme; small groups need caution")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3)
    save(figure, "eda_05_courses")
    tables = ROOT / "artifacts/tables"
    tables.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(tables / "training_correlations.csv")
    course_counts.to_csv(tables / "training_course_outcomes.csv")
    grouped = training.groupby("Target", observed=True).agg(
        n=("Target", "size"), mean_age=("Age at enrollment", "mean"),
        mean_passed_sem1=("Curricular units 1st sem (approved)", "mean"),
        mean_grade_sem1=("Curricular units 1st sem (grade)", "mean"),
    ).reindex(CLASSES)
    grouped.to_csv(tables / "training_class_profiles.csv")
    return {"training_rows": len(training), "class_counts": {c: int(counts[c]) for c in CLASSES},
            "mean_passed_sem1": {c: float(grouped.loc[c, "mean_passed_sem1"]) for c in CLASSES},
            "age_range": [int(training["Age at enrollment"].min()), int(training["Age at enrollment"].max())],
            "passed_grade_correlation": float(correlations.loc["Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"]),
            "plot_count": 5}


def result_figures(summary: dict) -> None:
    """Read artifacts only: regenerating figures never fits or selects a model."""
    tables = ROOT / "artifacts/tables"
    history = pd.read_csv(tables / "ann_history.csv")
    history = history.loc[history.candidate == summary["ann"]["selected_candidate"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for column, label, color in (("train_log_loss", "Train", "#277f87"), ("validation_log_loss", "Validation", "#c84b53")):
        axes[0].plot(history.epoch, history[column], label=label, color=color)
    for column, label, color in (("train_macro_f1", "Train", "#277f87"), ("validation_macro_f1", "Validation", "#c84b53")):
        axes[1].plot(history.epoch, history[column], label=label, color=color)
    for axis in axes:
        axis.axvline(summary["ann"]["best_epoch"], color="#6c7078", linestyle="--", label="Restored epoch")
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
    axes[0].set(ylabel="Unweighted cross-entropy", title="ANN loss on original class distribution")
    axes[1].set(ylabel="Macro F1", title="ANN convergence and overfitting")
    figure.tight_layout()
    save(figure, "ann_training_history")

    comparison = pd.read_csv(tables / "final_comparison.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comparison))
    axis.bar(x - .18, comparison.validation_macro_f1, width=.36, color="#277f87", label="Validation")
    axis.bar(x + .18, comparison.test_macro_f1, width=.36, color="#6a79a9", label="Test")
    axis.set_xticks(x, comparison.model, rotation=15, ha="right")
    axis.set(ylim=(0, 1), ylabel="Macro F1", title="Same split and feature profile for all main models")
    axis.legend()
    save(figure, "model_comparison")

    confusion = pd.read_csv(tables / "selected_confusion_matrix.csv", index_col=0)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, normalise in zip(axes, (False, True)):
        values = confusion.to_numpy(dtype=float)
        if normalise:
            values /= values.sum(axis=1, keepdims=True)
        axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalise else None)
        axis.set_xticks(range(3), CLASSES, rotation=25, ha="right")
        axis.set_yticks(range(3), CLASSES)
        axis.set(xlabel="Predicted", ylabel="Actual", title="Within actual class" if normalise else "Student counts")
        for i in range(3):
            for j in range(3):
                value = values[i, j]
                axis.text(j, i, f"{value:.1%}" if normalise else f"{int(value)}", ha="center", va="center",
                          color="white" if value > values.max() * .55 else "#202020")
    figure.suptitle(f"Frozen selected model: {summary['selected_model']}", y=1.03)
    figure.tight_layout()
    save(figure, "selected_confusion_matrix")

    selection = pd.read_csv(tables / "cluster_selection.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(selection.k, selection.inertia, "o-", color="#277f87")
    axes[0].set(xlabel="k", ylabel="Inertia", title="Elbow diagnostic (training only)")
    axes[1].plot(selection.k, selection.silhouette, "o-", color="#6a79a9")
    ineligible = selection.loc[selection.minimum_cluster_fraction < .05]
    if len(ineligible):
        axes[1].scatter(ineligible.k, ineligible.silhouette, marker="x", s=90, linewidths=2,
                        color="#c84b53", label="A cluster has <5% of students", zorder=4)
    axes[1].axvline(summary["unsupervised"]["k"], linestyle="--", color="#c84b53", label="Selected feasible k")
    axes[1].set(xlabel="k", ylabel="Silhouette", title="Selected k and separation")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(selection.k)
    figure.tight_layout()
    save(figure, "cluster_selection")

    projected = pd.read_csv(tables / "training_pca_projection.csv")
    variance = pd.read_csv(tables / "pca_variance.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in sorted(projected.cluster.unique()):
        subset = projected.loc[projected.cluster == label]
        axes[0].scatter(subset.PC1, subset.PC2, s=9, alpha=.35, label=f"Cluster {label}", rasterized=True)
    for label, color in zip(CLASSES, COLORS):
        subset = projected.loc[projected.Target == label]
        axes[1].scatter(subset.PC1, subset.PC2, s=9, alpha=.35, label=label, color=color, rasterized=True)
    for axis in axes:
        axis.set(xlabel=f"PC1 ({variance.explained_variance_ratio.iloc[0]:.1%})",
                 ylabel=f"PC2 ({variance.explained_variance_ratio.iloc[1]:.1%})")
        axis.legend(fontsize=8)
    axes[0].set_title("K-Means groups in PCA space")
    axes[1].set_title("Outcomes shown after fitting and selecting clusters")
    figure.tight_layout()
    save(figure, "pca_clusters")

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(variance.component, variance.explained_variance_ratio, color="#6a79a9", label="Individual")
    axis.plot(variance.component, variance.cumulative_variance, "o-", color="#277f87", label="Cumulative")
    axis.axhline(.95, linestyle="--", color="#c84b53", label="95%")
    axis.set(xlabel="Principal component", ylabel="Explained variance ratio", ylim=(0, 1.05), title="PCA information retained")
    axis.legend()
    save(figure, "pca_variance")

    ablation = pd.read_csv(tables / "feature_ablation.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ablation))
    axis.bar(x - .18, ablation.validation_macro_f1, width=.36, color="#277f87", label="Validation")
    axis.bar(x + .18, ablation.test_macro_f1, width=.36, color="#6a79a9", label="Test")
    axis.set_xticks(x, ["No semester results", "+ Semester 1", "+ Semester 2"])
    axis.set(ylim=(0, 1), ylabel="Macro F1", title="Logistic regression: historical feature-group comparison")
    axis.legend()
    save(figure, "feature_ablation")
