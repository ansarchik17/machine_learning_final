# Student Outcome Classification and Academic Profile Analysis

## Abstract

This study classifies historical student outcomes using UCI dataset 697, with
4,424 observations, 36 original predictors and three classes: Dropout, Enrolled
and Graduate. The main comparison uses first-semester information: 28 original
and four engineered features become 229 inputs after preprocessing. The shared
stratified split contains 3,096 training, 664 validation and 664 test observations.
Learned transformations are fitted only on the relevant training partition,
including the training portion of each cross-validation fold.

The supervised comparison includes a Dummy baseline, Logistic Regression,
Decision Tree, KNN and a feedforward artificial neural network (ANN). The ANN is
selected by validation Macro F1 of 0.6625. Its test Macro F1 is 0.6785 and its
accuracy is 72.89%. Logistic Regression achieves a higher test Macro F1 of 0.6899;
this finding does not change the original model selection. A separate Logistic
Regression comparison gives test Macro F1 of 0.5683 without semester results,
0.6899 with the first semester and 0.7255 with both semesters. K-Means identifies
three academic profiles. Two PCA components retain 77.80% of the variance in the
five academic indicators examined.

These results show the value of academic information in this historical sample.
The absence of individual outcome dates and external validation limits any
interpretation as an operational early-warning system. The research proposal
sets out a further study of the stability of the comparisons.

## Introduction

Classifying educational outcomes may help a university identify students who
need additional review or support. This requires a clear outcome definition, an
account of the information available and an evaluation procedure. A student who
is still enrolled at the end of a programme's expected duration does not have a
known final unsuccessful outcome.

The project asks how classification performance changes when first- and
second-semester results are added. Two experiments address this question. The
first compares algorithms using the same first-semester feature profile. The
second compares three information profiles while keeping Logistic Regression
as the algorithm. Combining these comparisons would obscure the source of the
observed differences.

The practical objective is to improve Macro F1 over a most-frequent-class
baseline and describe the remaining errors. The experiment does not estimate
an intervention's effectiveness or the causes of dropout. The related work
connects this design to educational analytics and methodological research.

## Dataset and EDA

Each row represents one student. The current CSV contains 36 predictors and
Target. The preparation script downloads the official UCI archive, extracts the
CSV and retains its original bytes in data/raw/students_dropout.csv. The file
is loaded with pandas using a semicolon delimiter. A recorded SHA-256 checksum
checks the source version. The source describes one Portuguese institution;
the dataset's publication date is not the observation date of every student.
The bibliography and source metadata give the link and CC BY 4.0 licence.

No complete duplicate rows or empty cells were found, and no rows were removed.
Code 34 for unknown parental education and code 99 for unknown parental occupation
are mapped to Unknown. An extra tab character in one column heading is removed
in memory. Binary codes are checked and quantitative ranges are recorded in the
audit. Zero grades and valid extreme values are retained. Target is neither
imputed nor reduced to two classes. Individual event dates and an explicit
admission-year field are absent from the CSV.

The five EDA plots use training data only. Class imbalance motivates Macro F1.
The age distributions overlap, while passed units and mean grades differ across
outcomes. Correlations show partial redundancy among academic indicators.
Programme-level outcome proportions require attention to sample sizes. These
plots describe associations and do not establish causes.

## Methodology

All main models use the same 3,096 training rows and the same information. The
70/15/15 split is stratified with random seed 42, and row membership is saved
before modelling. Five cross-validation folds are formed within training data.
This random split does not simulate the arrival of later admission cohorts.

The main profile excludes second-semester indicators and two financial statuses
whose individual observation dates are unknown: Debtor and Tuition fees up to
date. This leaves 28 original predictors. Four first-semester features are added:
pass_rate, evaluations_per_unit, without_evaluations_rate and zero_enrollment.
Ratios with a zero denominator are set to zero and retain a separate indicator.
Categorical codes receive one-hot encoding. Quantitative inputs are standardised
for LR, KNN and ANN; the tree does not require scaling. The Pipeline includes a
median imputer, although no source cells require filling. Encoder, imputer and
scaler are fitted on the relevant training portion, including inside each CV fold.

Dummy predicts the most frequent training class. Logistic Regression tests C
values of 0.1, 1 and 10 with ordinary or balanced class weights. The tree tests
maximum depths of 3, 5, 8 and unrestricted, minimum leaf sizes of 5 and 15, and
two weighting settings. KNN tests 5, 15 and 31 neighbours with uniform or distance
weights. These give 6, 16 and 6 configurations. The small grids explore
regularisation, tree complexity and neighbourhood size while keeping the
computation manageable. Each algorithm's candidate is chosen by mean training-CV
Macro F1. The selected candidates are then compared on the shared validation set.

The ANN search contains four configurations declared before evaluation. The
selected network has 229 inputs, 32 hidden ReLU units and three softmax outputs,
with 7,459 parameters. It uses cross-entropy with L2 alpha 0.01 and class weights
from training data. Adam uses learning rate 0.001, beta1 0.9, beta2 0.999 and
epsilon 1e-8, with batch size 64. Training allows at most 200 epochs with patience
25. The checkpoint maximises validation Macro F1; an exact tie favours lower
validation cross-entropy. The main model also maximises validation Macro F1;
exact ties retain the declared candidate order.

Settings and models are saved before test metrics are calculated. There is no
refit on combined training and validation data. Classical cross-validation and
ANN validation tuning have different search budgets, although the final data
and test protocol are shared. Repeated validation use remains a limitation of
model selection.

Macro F1 gives equal weight to the F1 of each class. Additional metrics are
Dropout precision and recall, balanced accuracy, accuracy and the confusion
matrix. Class weights address imbalance without generating synthetic students.

## Model results and ANN

All main models exceed Dummy on test Macro F1. Validation selects the ANN,
while Logistic Regression has a slightly higher test Macro F1. Both findings
are reported without revising the original choice. Macro F1 and accuracy
measure different properties: KNN has test accuracy of 70.33%, but its Enrolled
recall is only 18.49%.

The selected ANN's best epoch is 9 and training stops at epoch 34. Training
Macro F1 increases from 0.7077 at the selected epoch to 0.7913 at the end, while
validation Macro F1 decreases from 0.6625 to 0.6324. This pattern is consistent
with overfitting and supports restoring the earlier checkpoint. The displayed
cross-entropy curves are unweighted; optimisation also used class weights and L2.

## Feature-group comparison

Logistic Regression is evaluated on three profiles with the same rows, grid and
CV procedure. The profiles contain 22, 32 and 42 features before one-hot encoding.
Adding the first semester gives the main test Macro F1 increase, from 0.5683 to
0.6899; both semesters give 0.7255. This is a historical comparison of information.
The two-semester model is not ranked against the main ANN as if only the
algorithm differed.

## Clustering and PCA

K-Means uses five quantitative first-semester indicators: enrolled units,
evaluations, approved units, mean grade and units without evaluations. Credited
units are excluded because they represent recognition of previous learning.
StandardScaler is fitted on training data. The search tests k=2..6, n_init=20
and seeds 42, 43 and 44. Selection maximises silhouette subject to a minimum
cluster share of 5%; ties favour smaller k. Target is added only after selection
to describe the groups.

The selected k is 3, with silhouette 0.6002 and mean adjusted Rand index of 1.0
across the tested initialisations. At k=4, silhouette is higher, but a 3.10%
group violates the declared size constraint. The profiles describe ordinary
load with progress, almost no passed units and a higher registered load. Their
sizes are 2,401, 512 and 183. Outcome shares are descriptive, rather than
individual predictions or evidence of causes.

PCA examines the same five standardised training indicators. Two components
retain 77.80% of variance and four retain 97.92%. A two-dimensional plot discards
the remaining variation. PCA does not replace the main ANN's 229 inputs. High
stability across the tested seeds does not establish stability in new populations.

## Error analysis and discussion

The ANN correctly classifies 484 of 664 test observations. Dropout recall is
155/214 = 72.43% and precision is 155/200 = 77.50%. Enrolled recall is
61/119 = 51.26%, and Graduate recall is 268/331 = 80.97%. Among actual Dropout
students, 20 are predicted as Graduate and 39 as Enrolled. Enrolled is the
most difficult class for this model. Heterogeneous trajectories among students
who are still studying offer a possible explanation, rather than an established
mechanism.

Descriptive slices by age, sex, scholarship status and zero academic progress
are saved with group sizes. They do not demonstrate fairness. Validation
permutation importance ranks pass_rate, approved units, grade, Course and
evaluations_per_unit highest. Correlated and engineered features limit
individual attribution: permuting one may disrupt their joint consistency.

The main limitations are one institution, historical cohorts, one main split,
missing individual dropout dates and incomplete knowledge of when changing
statuses were observed. Excluding two financial columns does not establish the
absence of all temporal leakage. The scores do not demonstrate deployment
readiness, causal factors or the effectiveness of an educational intervention.
A possible application is prioritising supportive human review after further
validation. A missed Dropout could mean missed support, while a false warning
would require additional review and could impose an unfair burden.

The source file, split, CV folds, parameters, versions, models and predictions
are saved. Repeating the unchanged experiment reproduced its metrics and
checkpoint. This checks reproducibility on the same test set, not external
validity. Actual technical contributions must be supported by individual records.

## Conclusion

Under the fixed historical protocol, the models use informative student data
and improve on the most-frequent-class baseline. Validation selects the ANN,
while LR scores higher on test; the original selection protocol is preserved.
Adding academic results improves Logistic Regression and answers the project's
question about the value of additional information in this sample.

K-Means and PCA describe academic structure and complement classification.
Further work should examine comparison stability and the availability of
features over time. The accompanying research proposal states hypotheses,
evaluation procedures, limitations and a work plan.

## AI assistance

Codex assisted with code preparation, computational checks, primary-source
searches, draft writing, English translation and presentation preparation.
Numerical results come from saved experiment outputs. Bibliographic details
were checked against primary sources and publisher metadata. This is an
AI-assisted draft for team review. Each member must verify the claims, understand
and explain the code and results, and record their actual technical contribution.
The proposed allocation of work is not a record of work already completed.
