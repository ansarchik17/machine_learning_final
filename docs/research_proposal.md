# Stability of Student Outcome Classification

## Research question and hypotheses

The completed project found better historical classification when semester
results were added, but it used one stratified split. Studies of temporal
variation show that the value of information depends on the stage of study
(Glandorf et al., 2024; Vaarma & Li, 2024). The next step is to examine comparison
stability while retaining three outcomes and separating information availability
from algorithm choice.

Research question: do the benefit of academic features and the ranking of ANN
and Logistic Regression persist under repeated evaluation of historical
observations and restricted use of sociodemographic information?

H1. Adding first-semester results increases mean Macro F1 over the profile
without semester results in each of three prespecified outer-evaluation repeats.
The comparison uses Logistic Regression and identical outer folds.

H2. Removing the sociodemographic block reduces mean Macro F1 by no more than
0.03 compared with the full first-semester profile. This is an exploratory
margin chosen before the new runs, not an established threshold of practical
equivalence. Results that contradict the hypothesis will also be reported.

H3. The ordering of ANN and Logistic Regression varies across splits. The study
examines the sign and spread of their paired Macro F1 differences without
assuming that ANN must perform better.

The objective is to assess the stability of conclusions and sensitivity to the
information used. The study will not estimate the causal effect of student
support or establish readiness for automated educational decisions. Retaining
multiple outcomes and reporting Enrolled metrics addresses the heterogeneous
goals discussed in the literature (Setiawan et al., 2026).

<!-- pagebreak -->

## Data and experimental design

The study will use the same 4,424 real observations from UCI 697 and its Dropout,
Enrolled and Graduate classes (Realinho et al., 2022). The original CSV and
checksum will be retained. No synthetic observations or artificial Target
filling will be used. The current test set has already been exposed. New scores
will therefore be interpreted as repeated internal analysis of the historical
dataset, rather than independent external validation.

Three information profiles will be fixed: no semester results, the first
semester and both semesters. A separate first-semester variant will exclude
age, sex, marital status, nationality, international status and parental
characteristics. The exact block will be recorded before running the experiments.
Debtor and Tuition fees up to date will remain excluded from all core variants.
Other changing statuses will be treated as an availability limitation, rather
than assumed to have been known at admission.

Outer evaluation will use stratified five-fold splits with seeds 42, 2026 and
2027. Within each outer training portion, a shared 20% validation subset will
be reserved for both candidates; the other 80% will be the effective training
set. Classical parameters will be selected with three-fold CV inside that
training set. ANN will use the same training and validation data to choose its
configuration and checkpoint. Both models will be evaluated on the same outer
held-out portion. Encoder, scaler and imputer will be fitted only on the relevant
training subsets.

The limited LR grids and four ANN configurations declared in the current
project will be retained. Selection criteria, tie-breaking rules, seeds and
early stopping will be specified before the new experiments. This separates
outer performance estimation from tuning and addresses selection overfitting
(Cawley & Talbot, 2010).

Macro F1 will be the primary metric. Additional metrics will include Dropout
precision and recall, per-class F1, balanced accuracy and confusion matrices.
Comparisons will be paired on identical outer folds. Clustering and PCA will
remain descriptive and will not select the outcome classifier.

<!-- pagebreak -->

## Risk analysis and work plan

For each repeat, the study will show outer-fold scores and paired differences
between profiles and models, followed by averages, ranges and variation across
repeats. Dependent folds will not be treated as independent observations in an
ordinary t-test. Group metrics will include group sizes and event counts.
Removing demographic features alone does not establish fairness.

The four-week plan assigns week one to fixing hypotheses and implementing the
splits, week two to reproducible experiments, week three to error and stability
checks, and week four to interpretation and reporting. The existing Python
environment and CPU will be used. Scope is limited to one dataset, two algorithms
and small search grids. New results will be saved separately from the current
frozen experiment, which will not be retuned.

Expected outputs are an open experimental specification, split and fold
manifests, comparison tables and an account of limitations. A single institution
and missing individual time stamps will continue to limit transferability.
Validation on an external cohort with known observation dates is a later
research direction, rather than a promised outcome of the four-week study.
Possible use concerns student support with human review.

## Proposal references

This proposal uses Cawley and Talbot (2010), Glandorf et al. (2024), Realinho et al.
(2022), Setiawan et al. (2026), and Vaarma and Li (2024). Their complete entries
are included in the proposal PDF from the verified reference registry and in
the project's full bibliography. Codex assisted with drafting and translation;
the team must review the proposal before submission.
