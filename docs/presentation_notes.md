# Defense Speaking Notes

Target duration: 12 minutes, 15 slides. Speaking assignments are proposed for rehearsal and do not confirm technical contributions. Timings include pauses and transitions. Do not read the source lists or Q&A aloud.

## 1. Student Outcome Classification and Academic Profile Analysis

Suggested speaker: Ansar Keles. Time: 25 seconds.

Good afternoon. Our project classifies student outcomes using real UCI data with three classes: Dropout, Enrolled and Graduate. We ask how semester results change classification performance. We first compare algorithms on the same information, then examine semester features separately. We finish with academic profiles, errors and limitations.

Sources: project_config.json, docs/report.md.

## 2. Research question

Suggested speaker: Beknazar Sarsenbek. Time: 55 seconds.

How does classification performance change when first- and second-semester results are added? We separate two comparisons. The first keeps rows and features fixed while changing the algorithm. The second keeps Logistic Regression as the algorithm while changing the available information. The literature motivates this distinction. Glandorf and colleagues examine how information value varies over time. Cawley and Talbot explain how model selection can overfit. We therefore do not assume that a more complex model must be better. We retain three outcomes and examine performance for each class. The next slide introduces the dataset and its outcome definitions.

Sources: docs/literature_review.md, https://doi.org/10.1145/3636555.3636906, https://jmlr.org/papers/v11/cawley10a.html.

## 3. Dataset and three outcomes

Suggested speaker: Mukhamedzhan Sagym. Time: 45 seconds.

The dataset contains 4,424 students, 36 original predictors and Target. These are historical observations from one Portuguese institution. Graduate represents almost half the sample, whereas Enrolled represents about eighteen percent. Enrolled means that the student was still studying at the end of the expected programme duration. It is not a known final outcome. Dropout includes leaving the programme and may include a transfer elsewhere. We retain all three classes. This slide uses the entire dataset; subsequent EDA uses training data only. UCI provides the source under the CC BY 4.0 licence.

Sources: data/source_metadata.json, project_config.json, https://doi.org/10.24432/C5MC89, https://doi.org/10.3390/data7110146.

## 4. Evaluation protocol

Suggested speaker: Mukhamedzhan Sagym. Time: 60 seconds.

We use one shared stratified split: seventy percent training, fifteen percent validation and fifteen percent test, with seed 42. This gives 3,096, 664 and 664 observations. All main models use the same rows. Classical hyperparameters are chosen through five-fold cross-validation within training data. Each fold fits its encoder, imputer and scaler using only that fold's training portion. Validation compares candidates and chooses the ANN configuration and checkpoint. Test metrics are computed after models are frozen. Test results do not feed back into tuning, and there is no combined training-plus-validation refit. Repeating the unchanged experiment checked reproducibility, rather than providing independent external validation.

Sources: project_config.json, artifacts/experiment_plan.json, artifacts/frozen_models.json, artifacts/tables/split_counts.csv.

## 5. Features and preprocessing

Suggested speaker: Mukhamedzhan Sagym. Time: 45 seconds.

The main profile uses the first semester. We exclude second-semester indicators and two financial statuses with unknown individual observation dates. That leaves 28 original predictors. Four engineered features describe the pass rate, evaluations per unit, units without evaluation and zero enrolment. One-hot encoding expands the 32 features to 229 inputs. Numerical category codes remain categorical. Special unknown parental values become Unknown. There are no empty cells or complete duplicate rows, and zero values are retained. Learned transformations are fitted only on training data. These steps do not establish when every changing characteristic was actually observed.

Sources: src/data.py, artifacts/data_audit.json, artifacts/tables/cleaning_log.csv, artifacts/tables/feature_dictionary.csv.

## 6. Academic progress in training data

Suggested speaker: Mukhamedzhan Sagym. Time: 40 seconds.

This chart shows mean passed first-semester units in training data only. The means are 2.58 for Dropout, 4.40 for Enrolled and 6.17 for Graduate. This association motivates examining semester features, but the means do not describe every student or establish a causal effect. The distributions overlap. The full notebook and report also show age distributions, correlations and programme-level differences. Test data are not used to choose features from these plots. Next, Beknazar explains the classical models and our main evaluation metric.

Sources: artifacts/tables/training_class_profiles.csv, notebooks/students_dropout_project.ipynb.

## 7. Classical models and Macro F1

Suggested speaker: Beknazar Sarsenbek. Time: 50 seconds.

Dummy predicts the most frequent training class and supplies a reference baseline. Logistic Regression, Decision Tree and KNN represent linear relationships, rules and local similarity. We evaluate six, sixteen and six configurations respectively. The table reports their selected parameters and mean Macro F1 across five training folds. LR reaches 0.648, the tree 0.611 and KNN 0.561. Macro F1 gives equal weight to each class's F1. Accuracy can hide weak minority-class performance: KNN reaches about seventy percent test accuracy, but only 18.49 percent Enrolled recall. The CV values in this table are not the final test results.

Sources: artifacts/tables/cv_logistic_main.csv, artifacts/tables/cv_tree_main.csv, artifacts/tables/cv_knn_main.csv, artifacts/tables/test_class_metrics.csv.

## 8. ANN architecture

Suggested speaker: Ansar Keles. Time: 50 seconds.

The selected network has 229 prepared inputs, one hidden layer of 32 ReLU units and three softmax outputs. It contains 7,459 trainable parameters. Training uses cross-entropy, L2 regularisation and class weights calculated from training data. Adam uses learning rate 0.001 and batch size 64. Validation chooses this weighted network from four configurations declared before evaluation. Training allows at most 200 epochs with patience 25. The checkpoint maximises validation Macro F1, with lower validation cross-entropy breaking an exact tie. The next slide shows the selected network's actual saved training history.

Sources: artifacts/results_summary.json, artifacts/tables/ann_candidates.csv, src/workflow.py, https://arxiv.org/abs/1412.6980.

## 9. Early checkpoint preserves validation F1

Suggested speaker: Ansar Keles. Time: 50 seconds.

The lines show all 34 recorded epochs of the selected ANN. The Macro F1 axis runs from 0.50 to 0.85 to make the changes visible. The best checkpoint is epoch nine, with training F1 around 0.708 and validation around 0.662. By epoch 34, training rises to 0.791 while validation falls to 0.632. This is consistent with overfitting and explains why earlier weights are restored. Loss shows a similar pattern: training cross-entropy falls while validation loss rises. Those saved loss curves are unweighted, although optimisation also used class weights and L2. The restored model supplies the final test predictions.

Sources: artifacts/tables/ann_history.csv, artifacts/results_summary.json.

## 10. Validation selects ANN, LR scores higher on test

Suggested speaker: Ansar Keles. Time: 60 seconds.

This is the main comparison on identical first-semester information. All Macro F1 bars use a zero-to-one scale. ANN has the highest validation score, 0.6625, so the declared procedure selects it. On test, Logistic Regression reaches 0.6899 and ANN reaches 0.6785. We preserve the original selection and report the different ordering. A difference on one split does not establish statistical significance or a stable ranking. Every main model exceeds Dummy. LR test accuracy is 73.04 percent and ANN accuracy is 72.89 percent, but accuracy is a separate metric. We now turn to an experiment that changes the information profile while keeping the algorithm fixed.

Sources: artifacts/tables/final_comparison.csv, artifacts/frozen_models.json, artifacts/verification.json.

## 11. First-semester information improves Macro F1

Suggested speaker: Beknazar Sarsenbek. Time: 50 seconds.

All three profiles here use Logistic Regression with the same rows, grid and CV procedure. Test Macro F1 is 0.5683 without semester results, 0.6899 with the first semester and 0.7255 with both. The first-semester gain is about 0.122 and the additional second-semester gain is about 0.036 Macro F1 points. Most of the improvement in this comparison comes from the first semester. These are not percentage gains in accuracy. The two-semester model has more information than our main ANN. Missing individual outcome dates also mean that this historical comparison does not demonstrate an operational early-warning system at a particular date.

Sources: artifacts/tables/feature_ablation.csv, artifacts/experiment_plan.json.

## 12. Academic profiles and PCA

Suggested speaker: Mukhamedzhan Sagym. Time: 65 seconds.

The exploratory branch uses five quantitative first-semester indicators, standardised on training data. Target is excluded from K-Means fitting and selection. We examine k from two to six and maximise silhouette subject to a minimum cluster size of five percent. This selects three clusters. Four has a slightly higher silhouette, but its smallest group is only 3.10 percent and fails the size condition. The profiles describe ordinary load, almost no passed units and higher registered load. They are descriptive groups. PCA examines the same five indicators. Two components retain 77.80 percent of variance and four retain 97.92 percent. PCA does not replace ANN inputs, and stability across these seeds does not establish transferability to another institution.

Sources: artifacts/tables/cluster_selection.csv, artifacts/tables/cluster_profiles.csv, artifacts/tables/pca_variance.csv, artifacts/models/unsupervised.joblib.

## 13. Enrolled is the most difficult class

Suggested speaker: Ansar Keles. Time: 55 seconds.

The confusion matrix shows the selected ANN on 664 test students. Rows are actual outcomes and columns are predictions. The diagonal contains 484 correct classifications, leaving 180 errors. Of 214 actual Dropout students, 155 are identified, giving recall of 72.43 percent. Another 39 are predicted as Enrolled and 20 as Graduate. Enrolled is the hardest class: only 61 of 119 are correctly classified, giving recall of 51.26 percent. Heterogeneous study trajectories are a possible explanation, rather than a demonstrated cause. In a support application, a missed Dropout could mean missed support, while a false warning requires review. Descriptive group slices do not establish fairness.

Sources: artifacts/tables/selected_confusion_matrix.csv, artifacts/tables/test_class_metrics.csv, artifacts/tables/test_error_groups.csv.

## 14. Conclusions and further research

Suggested speaker: Beknazar Sarsenbek. Time: 45 seconds.

Academic information improves Logistic Regression in this historical sample. A more complex model does not necessarily lead on test. The main limitations are one institution, one main split and missing individual dates for outcomes and some features. The proposal describes future repeated outer folds with three seeds, testing the stability of ANN versus LR and sensitivity to excluding demographic features. These experiments have not yet been performed. The already exposed test limits the independence of further analysis of the same dataset. Before a support application, evaluation would require an external cohort with known observation dates and human review.

Sources: docs/research_proposal.md, docs/report.md, artifacts/tables/feature_ablation.csv.

## 15. References and questions

Suggested speaker: Beknazar Sarsenbek. Time: 25 seconds.

These are selected dataset and methodological references. The report contains the full 26-source APA bibliography. Our experimental metrics come from saved project outputs. Codex assisted with code, source searches, drafting and translation. The team must review the material and explain the methods and results. Thank you. We are ready for questions.

Sources: docs/bibliography.md, docs/research/sources.json, docs/contributions.csv.

## Defense Questions

**Why Macro F1?**

The classes are imbalanced. Macro F1 weights each class's F1 equally, whereas accuracy can hide poor Enrolled recall.

**Why select ANN when LR scores higher on test?**

The declared procedure selects on validation. Test is exposed only after freezing. LR's higher test score is reported without revising the selection.

**Can LR 0.7255 be ranked against ANN 0.6785?**

Not as an algorithm-only comparison. That LR uses both semesters and the main ANN uses only the first. The main algorithm comparison holds information fixed.

**How do 32 features become 229 inputs?**

One-hot encoding expands categorical columns. Before encoding there are 28 original and four engineered features. The encoder learns categories only from training data.

**Why restore epoch 9?**

It gives the selected candidate's best validation Macro F1. Training stops at 34 after patience 25. Later training performance is higher, but validation is lower.

**Why k=3 when k=4 has higher silhouette?**

Selection also requires every group to contain at least 5% of training rows. The minimum is 3.10% for k=4 and 5.91% for k=3.

**Did PCA improve ANN classification?**

That experiment was not performed. PCA describes five academic indicators, while ANN uses 229 prepared inputs. Explained variance is not classification performance.

**Does excluding financial statuses remove all leakage?**

No. Preprocessing is fitted on training data, but individual outcome dates and observation times for some features are missing. Temporal availability remains a limitation.

**What does the repeated training run establish?**

Reproducibility of the frozen code and plan on the same split. It is not independent evaluation or evidence of transfer to another institution.

**Who completed each part?**

Speaking assignments and technical scopes are proposed. Actual contributions require individual evidence and the contribution record. AI assistance is disclosed.
