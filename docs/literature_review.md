# Literature Review: Student Outcomes

## Educational outcomes and available information

Predicting an educational outcome depends on its definition, the observation
point and the features available. A learning-analytics review links prediction
to academic performance and learning activity, but covers a wider range of tasks
than programme dropout [(Namoun & Alshanqiti, 2021)](https://doi.org/10.3390/app11010237).
Predicting a grade, completing an individual course and leaving a university
programme therefore require different interpretations of performance.

This project uses the published UCI dataset
[(Realinho et al., 2021)](https://doi.org/10.24432/C5MC89). Its accompanying paper
defines the three classes relative to the expected duration of the programme.
Leaving includes changing programme or institution. Enrolled is consequently
retained as a distinct observed status, while Dropout is not interpreted solely
as academic dismissal [(Realinho et al., 2022)](https://doi.org/10.3390/data7110146).

A recent review of multiclass educational prediction describes the predominance
of binary tasks and variation in imbalance handling, metrics and validation.
Per-class metrics and Macro F1 provide more detail than accuracy alone
[(Setiawan et al., 2026)](https://doi.org/10.7160/eriesj.2026.190106). This supports
our three-class formulation, the primary use of Macro F1 and a separate
examination of Dropout precision and recall.

## Accumulating academic information

A fair model comparison requires the same information. Kemper et al. examine
logistic regression and trees for predicting dropout from university data
[(Kemper et al., 2020)](https://doi.org/10.1080/21568235.2020.1718520). These methods
provide a useful basis for comparison with a neural network. Model complexity
alone does not establish better performance on a new sample.

Vaarma and Li compare academic, demographic and learning-management-system
information, examining how feature importance and performance vary by month
[(Vaarma & Li, 2024)](https://doi.org/10.1016/j.techsoc.2024.102474). Glandorf et al.
consider temporal and between-group variation and show how the relevance of
university academic information increases during study
[(Glandorf et al., 2024)](https://doi.org/10.1145/3636555.3636906). Together, these
studies motivate examining information availability separately from algorithm
selection.

In this project, Logistic Regression is compared across profiles without
semester results, with the first semester and with both semesters. Shared rows,
a parameter grid and CV allow us to examine the contribution of additional
information. Publications using LMS logs, different outcomes or other cohorts
are not numerical benchmarks for our Macro F1. Their targets, populations,
prediction points and evaluation designs differ.

## Validation and information leakage

Leakage includes using information that should not be available in the intended
prediction setting [(Kaufman et al., 2012)](https://doi.org/10.1145/2382577.2382579).
Fitting preprocessing only on training data addresses the statistical part of
this problem, but does not establish the temporal availability of every column.
Our CSV has no individual dropout dates. Two financial statuses are excluded
from the core profiles because their observation dates are uncertain. The
semester comparison remains a historical analysis of information.

Model selection can itself overfit a finite validation sample. Differences
between algorithms should be considered in light of the selection procedure
[(Cawley & Talbot, 2010)](https://jmlr.org/papers/v11/cawley10a.html). Our project
places preprocessing inside a Pipeline and selects classical parameters within
training CV. ANN uses validation for checkpoint selection and comparison of
four configurations. These procedural differences are acknowledged in the
interpretation. The scikit-learn guide describes separating fitting from
transformation and applying Pipelines within CV
[(Scikit-learn developers, n.d.)](https://scikit-learn.org/1.9/user_guide.html).

The main model is chosen on validation before test scores are computed. LR's
higher test score is reported without changing the original ANN selection.
Repeating the same pipeline checks reproducibility, but cannot replace a new
temporal or external evaluation. A small difference on one split does not
establish a stable algorithm ranking for other student populations.

## Classical models and a neural network

The baseline algorithms represent different relationships between features and
class: a linear probabilistic model, a tree of conditions and local neighbour
voting. Cover and Hart provide a foundational account of nearest-neighbour
classification [(Cover & Hart, 1967)](https://doi.org/10.1109/TIT.1967.1053964).
Their theoretical results do not guarantee the performance of our finite KNN
configuration on mixed features after encoding.

Backpropagation allows neural networks to learn hidden representations
[(Rumelhart et al., 1986)](https://doi.org/10.1038/323533a0). Adam uses adaptive
estimates of gradient moments [(Kingma & Ba, 2015)](https://arxiv.org/abs/1412.6980).
The project ANN uses ReLU, softmax, cross-entropy, L2 and class weighting. Its
width, learning rate and checkpoint are chosen through the declared experiment.
The original method papers do not determine an optimal configuration for this
dataset.

The observed learning curves show improving training performance alongside
worsening validation after an early checkpoint. This supports restoring the
saved epoch and comparing ANN with regularised Logistic Regression. A nonlinear
hidden layer does not guarantee an advantage.

## Clustering and PCA as additional analysis

Iterative assignment to centres is associated with minimising squared error
[(Lloyd, 1982)](https://doi.org/10.1109/TIT.1982.1056489). Our K-Means analysis uses
standardised training academic indicators without Target. Silhouette assesses
within-group cohesion and separation
[(Rousseeuw, 1987)](https://doi.org/10.1016/0377-0427(87)90125-7). Adjusted agreement
between partitions helps assess stability under different initialisations
[(Hubert & Arabie, 1985)](https://doi.org/10.1007/BF01908075).

Group size, silhouette and ARI answer different questions. Stable centres in
one sample do not establish stability at another institution. Outcome shares
are added after cluster selection. Cluster numbers are not predicted outcome
classes and do not replace supervised evaluation.

PCA constructs uncorrelated directions that successively retain variance
[(Jolliffe & Cadima, 2016)](https://doi.org/10.1098/rsta.2015.0202). Here it is a
separate projection of five academic indicators. Explained variance describes
the retained feature structure; classification performance is measured
separately. The main ANN uses 229 prepared inputs, rather than replacing them
with PCA components.

## Position of this project

Work on temporal information and multiclass evaluation supports two distinct
questions: comparing algorithms on one profile and comparing information
profiles within one algorithm. Our [model results](../artifacts/tables/final_comparison.csv)
and [ablation results](../artifacts/tables/feature_ablation.csv) show the value of
semester information in this historical sample. They do not establish causes
of dropout, effects of an educational intervention or superiority over studies
using other data.

The course project's contribution is a reproducible comparison of these
questions using a shared split, documented tuning, saved ANN history and joint
interpretation of errors, clusters and PCA. Further research should establish
when information becomes available and evaluate new cohorts or another
institution. That extension requires individual temporal information absent
from the current table.

## Software and reproducibility

Algorithms, preprocessing and metrics use scikit-learn
[(Pedregosa et al., 2011)](https://jmlr.org/papers/v12/pedregosa11a.html). Numerical
arrays and computation use NumPy [(Harris et al., 2020)](https://doi.org/10.1038/s41586-020-2649-2)
and SciPy [(Virtanen et al., 2020)](https://doi.org/10.1038/s41592-019-0686-2).
pandas processes tables [(McKinney, 2010)](https://doi.org/10.25080/Majora-92bf1922-00a),
and Matplotlib produces plots [(Hunter, 2007)](https://doi.org/10.1109/MCSE.2007.55).
Installed versions are recorded separately from the publication dates of these
foundational software papers.

The notebook combines code, results and explanations
[(Kluyver et al., 2016)](https://doi.org/10.3233/978-1-61499-649-1-87). Its format is
described in the nbformat documentation
[(Project Jupyter, n.d.-a)](https://nbformat.readthedocs.io/en/latest/), execution
uses nbclient [(Project Jupyter, n.d.-b)](https://nbclient.readthedocs.io/en/latest/),
and models are saved with Joblib [(Joblib developers, n.d.)](https://joblib.readthedocs.io/en/stable/).
These references describe implementation. The project's own saved artifacts
remain the source of its numerical results.

Full reference entries appear in the [bibliography](bibliography.md).
