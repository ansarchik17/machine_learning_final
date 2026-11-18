"""Build the annotated notebook from the completed, frozen numerical study."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = json.loads((ROOT / "artifacts/results_summary.json").read_text())
    cells = []
    md = lambda value: cells.append(nbf.v4.new_markdown_cell(value))
    code = lambda value: cells.append(nbf.v4.new_code_cell(value))
    md("# Student Outcome Classification and Academic Profile Analysis\n\n"
       "**Команда:** Ансар Келес, Мухамеджан Сагым, Бекназар Сарсенбек.\n\n"
       "Реальный датасет UCI 697. Три класса: Dropout, Enrolled, Graduate. "
       "Исследуем, как результаты семестров меняют качество исторической классификации. "
       "Код и черновики подготовлены с помощью Codex; фактический технический вклад студентов "
       "фиксируется отдельно в `docs/contributions.csv`.")
    md("## 1. Окружение и порядок запуска\n\n"
       "Запускайте все ячейки последовательно в ядре **Python (Students Dropout)**. "
       "При отсутствии результатов модуль выполнит весь объявленный эксперимент. "
       "При наличии результатов проверяются хеши и используются зафиксированные артефакты. "
       "Полное повторение того же плана доступно через `scripts/run_project.py --reproduce`; "
       "повторно используемый test не становится новой независимой выборкой.")
    code("from pathlib import Path\nimport sys, json, random\nimport numpy as np\nimport pandas as pd\n"
         "from IPython.display import display, Image\n"
         "ROOT = Path.cwd()\nif not (ROOT / 'project_config.json').exists():\n    ROOT = ROOT.parent\n"
         "assert (ROOT / 'project_config.json').exists(), 'Откройте notebook из папки проекта'\n"
         "sys.path.insert(0, str(ROOT))\nrandom.seed(42)\nnp.random.seed(42)\n"
         "from src.data import load_data, engineer_features, feature_profiles, fixed_split\n"
         "from src.workflow import run_experiments\n"
         "environment = json.loads((ROOT / 'artifacts/environment.json').read_text())\n"
         "print('Python:', sys.version.split()[0])\n"
         "display(pd.Series({k: v for k, v in environment['packages'].items() if k in "
         "['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn', 'nbformat', 'nbclient', 'ipykernel']}, name='version'))")
    md("Версии полного окружения находятся в `requirements.lock.txt`. Все модели используют один источник "
       "и фиксированные seed. Последующие ячейки не выбирают настройки по test.")
    md("## 2. Источник, очистка и feature engineering\n\n"
       "Исходный CSV сохраняется без изменения байтов и проверяется по SHA-256. "
       "Удаляется лишняя табуляция в заголовке; неизвестные значения образования/профессий родителей "
       "явно кодируются категорией Unknown. Нули не удаляются автоматически. "
       "Коды специальностей, гражданства и профессий являются категориями. "
       "Для каждого семестра добавлены доля сданных дисциплин, число оцениваний на дисциплину, "
       "доля дисциплин без оценивания и индикатор нулевой учебной нагрузки. "
       "При нулевом знаменателе ставка равна нулю, а отдельный индикатор сохраняет смысл случая.")
    code("raw_cleaned = load_data()\nframe = engineer_features(raw_cleaned)\nprofiles = feature_profiles(frame)\n"
         "audit = json.loads((ROOT / 'artifacts/data_audit.json').read_text())\n"
         "print('Строк:', len(frame), '| Исходных признаков:', raw_cleaned.shape[1] - 1)\n"
         "display(pd.Series(audit['encoded_unknown_counts'], name='coded_unknown_count'))\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/cleaning_log.csv'))\n"
         "display(pd.DataFrame([{'profile': name, 'features_after_engineering': len(columns)} for name, columns in profiles.items()]))")
    md("Target не заполняется и не используется в признаках. Debtor и Tuition fees up to date исключены "
       "из всех основных профилей из-за неопределённой даты наблюдения. При сравнении профилей "
       "не утверждается, что все остальные изменяемые статусы точно доступны при поступлении.")
    md("## 3. Разбиение и объявленный эксперимент\n\n"
       "Один stratified split 70/15/15, seed 42. Все кандидаты обучаются только на train; "
       "пять CV-fold находятся внутри train. One-hot encoder, median imputer и StandardScaler "
       "входят в Pipeline, поэтому при CV они переобучаются внутри каждого training fold. "
       "Настройки и лучший основной алгоритм выбираются без test.")
    code("splits, split_manifest = fixed_split(frame)\n"
         "summary = run_experiments(reuse=True)\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/split_counts.csv', index_col=0))\n"
         "print('Хеш плана:', summary['plan_sha256'])")
    md("План поиска записан до обучения в `artifacts/experiment_plan.json`. "
       "Настройки и выбранная модель сохранены в `artifacts/frozen_models.json` до расчёта test-метрик. "
       "Случайное разбиение оценивает историческую выборку и не имитирует новые годы поступления.")
    md("## 4. EDA: только тренировочные наблюдения")
    observations = [
        ("eda_01_class_balance", "Распределение исходов неравномерное. Macro F1 учитывает все три класса; "
         "одна accuracy может скрывать слабый результат на Enrolled."),
        ("eda_02_age", f"Возраст в train: {result['eda']['age_range'][0]}–{result['eda']['age_range'][1]} лет. "
         "Распределения перекрываются; возраст не определяет исход отдельного студента."),
        ("eda_03_passed_units", "Среднее число сданных дисциплин первого семестра в train: " + ", ".join(
            f"{name} — {value:.2f}" for name, value in result["eda"]["mean_passed_sem1"].items()) +
         ". Это ассоциация с будущим статусом, не доказательство причины."),
        ("eda_04_correlations", f"Корреляция количества сданных дисциплин со средней оценкой в train: "
         f"{result['eda']['passed_grade_correlation']:.3f}. Коррелированные академические показатели "
         "частично дублируют информацию; PCA ниже исследует эту структуру."),
        ("eda_05_courses", "Доли исходов различаются между программами. Сопоставляйте их с числом "
         "наблюдений; различия могут отражать состав студентов и особенности программы."),
    ]
    for name, interpretation in observations:
        code(f"display(Image(filename=str(ROOT / 'artifacts/figures/{name}.png')))")
        md(interpretation)
    md("## 5. Baseline, лекционные алгоритмы и hyperparameter tuning\n\n"
       "Baseline всегда предсказывает наиболее частый класс train. Logistic Regression использует "
       "C ∈ {0.1, 1, 10} и обычные/сбалансированные веса. Decision Tree проверяет глубины "
       "{3, 5, 8, без ограничения}, min_samples_leaf ∈ {5, 15} и веса классов. KNN проверяет "
       "k ∈ {5, 15, 31} и uniform/distance weights. Это ограниченный заранее объявленный поиск "
       "сложности, регуляризации и чувствительности к дисбалансу. Для каждого алгоритма лучший "
       "кандидат выбирается по среднему training-CV Macro F1. Масштабирование нужно LR/KNN/ANN; "
       "дерево обучается без него. Соответствие алгоритмов программе лекций проверяет команда.")
    code("for filename in ['cv_logistic_main.csv', 'cv_tree_main.csv', 'cv_knn_main.csv']:\n"
         "    cv_table = pd.read_csv(ROOT / 'artifacts/tables' / filename)\n"
         "    print(filename)\n"
         "    display(cv_table[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']].sort_values('rank_test_score').head(5))")
    md("Здесь mean_test_score означает отложенную часть CV-fold внутри train, а не финальный test. "
       "Среднее и разброс CV помогают оценить устойчивость результатов. Если train-score существенно "
       "выше CV-score, это признак возможного переобучения.")
    ann = result["ann"]
    architecture = " → ".join(map(str, [ann["input_dimension"], *ann["hidden"], 3]))
    md("## 6. Feedforward ANN\n\n"
       f"Выбрана архитектура **{architecture}**, {ann['parameter_count']:,} параметров. "
       "ReLU в скрытых слоях и softmax для трёх классов; cross-entropy, L2, Adam "
       "(learning rate 0.001, beta1=0.9, beta2=0.999, epsilon=1e-8), batch 64. "
       "Четыре кандидата сравнивают ширину, L2 и взвешивание классов. Максимум 200 эпох; "
       "patience 25 по validation Macro F1. Восстанавливается лучший checkpoint, "
       "при точном равенстве F1 — с меньшей validation cross-entropy. "
       "На графиках показаны невзвешенные loss, даже если оптимизация кандидата использовала веса.")
    code("display(pd.read_csv(ROOT / 'artifacts/tables/ann_candidates.csv'))\n"
         "display(Image(filename=str(ROOT / 'artifacts/figures/ann_training_history.png')))\n"
         "print('Восстановлена эпоха:', summary['ann']['best_epoch'], '| остановка:', summary['ann']['stopped_epoch'])")
    md("Снижение train loss при ухудшении validation loss указывает на переобучение. "
       "Кривые нужно интерпретировать вместе с выбранной эпохой и Macro F1. "
       "ANN использует те же train/validation/test и признаки, что основные классические модели; "
       "число кандидатов и способ настройки различаются и указаны явно.")
    history = pd.read_csv(ROOT / "artifacts/tables/ann_history.csv")
    chosen_history = history.loc[history.candidate == ann["selected_candidate"]]
    checkpoint = chosen_history.loc[chosen_history.epoch == ann["best_epoch"]].iloc[0]
    final_epoch = chosen_history.iloc[-1]
    md(f"У выбранной ANN на эпохе {ann['best_epoch']} train Macro F1={checkpoint.train_macro_f1:.3f}, "
       f"validation Macro F1={checkpoint.validation_macro_f1:.3f}. К последней эпохе train Macro F1 "
       f"вырастает до {final_epoch.train_macro_f1:.3f}, а validation становится "
       f"{final_epoch.validation_macro_f1:.3f}. Поэтому восстановлен ранний checkpoint; "
       "поздние эпохи не улучшили качество на validation.")
    unsup = result["unsupervised"]
    md("## 7. K-Means и интерпретация групп\n\n"
       "Используются пять количественных показателей первого семестра: enrolled, evaluations, "
       "approved, grade, without evaluations. Перезачёты исключены, поскольку описывают "
       "признание предыдущего обучения, а не только текущую активность. StandardScaler обучается "
       "на train. Проверяются k=2..6, n_init=20 и seed 42/43/44. Основной критерий — silhouette "
       "при минимальном размере группы 5%; ARI показывает устойчивость к инициализации. "
       "Target добавляется для описания только после выбора k.")
    code("display(pd.read_csv(ROOT / 'artifacts/tables/cluster_selection.csv').round(4))\n"
         "display(Image(filename=str(ROOT / 'artifacts/figures/cluster_selection.png')))\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/cluster_profiles.csv').round(3))")
    md(f"Выбрано **k={unsup['k']}**, silhouette={unsup['silhouette']:.3f}, mean seed ARI="
       f"{unsup['mean_seed_ari']:.3f}. Размеры групп и их академические профили приведены выше. "
       "Инициализационная устойчивость не доказывает существование естественных типов студентов. "
       "Кластеризация дополняет классификацию.")
    for group in unsup["cluster_profiles"]:
        enrolled = group["mean_Curricular units 1st sem (enrolled)"]
        approved = group["mean_Curricular units 1st sem (approved)"]
        grade = group["mean_Curricular units 1st sem (grade)"]
        description = ("почти отсутствуют успешно сданные дисциплины" if approved < .1 else
                       "повышенная зарегистрированная учебная нагрузка" if enrolled > 10 else
                       "обычная учебная нагрузка с выраженным академическим прогрессом")
        md(f"**Кластер {group['cluster']}: {description}.** n={group['n']}; среднее число "
           f"зарегистрированных дисциплин {enrolled:.2f}, успешно сданных {approved:.2f}, "
           f"средняя оценка {grade:.2f}. Доля Dropout={group['share_Dropout']:.1%}. "
           "Это описание исторических записей, а не диагноз или доказанная причина исхода.")
    md("Максимальная silhouette сама по себе не определяет выбор k: вариант, создающий группу "
       "меньше 5% train, исключается по заранее объявленному ограничению. Высокая нагрузка "
       "может отражать устройство программы; больше сданных дисциплин не обязательно означает "
       "более высокую среднюю оценку.")
    md("## 8. PCA\n\n"
       "PCA обучается на тех же стандартизованных тренировочных показателях. "
       "Это исследовательская проекция; основные supervised-модели не заменяют признаки на PCA-компоненты.")
    code("display(pd.read_csv(ROOT / 'artifacts/tables/pca_variance.csv').round(4))\n"
         "display(Image(filename=str(ROOT / 'artifacts/figures/pca_variance.png')))\n"
         "display(Image(filename=str(ROOT / 'artifacts/figures/pca_clusters.png')))\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/pca_loadings.csv', index_col=0).round(3))")
    md(f"Первые две компоненты сохраняют **{unsup['pc1_pc2_variance']:.2%}** дисперсии. "
       f"Для сохранения минимум 95% нужны **{unsup['components_for_95_percent']}** компоненты. "
       "Двумерная картинка теряет оставшуюся информацию; перекрытие групп и классов нельзя "
       "интерпретировать как отсутствие структуры во всех исходных измерениях.")
    md("## 9. Единое итоговое сравнение\n\n"
       f"Выбранная по validation модель: **{result['selected_model']}**. "
       "После заморозки настроек все модели оценены на общем test. "
       "Train+validation refit не выполнялся. Основная метрика — Macro F1; отдельно "
       "анализируются recall/precision Dropout, balanced accuracy и accuracy.")
    code("comparison = pd.read_csv(ROOT / 'artifacts/tables/final_comparison.csv')\n"
         "display(comparison.round(4))\n"
         "display(Image(filename=str(ROOT / 'artifacts/figures/model_comparison.png')))\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/test_class_metrics.csv').round(4))")
    md("Возможное отличие ранжирования на test не меняет модель, выбранную по validation. "
       "Значения test не используются для изменения параметров или признаков.")
    md("## 10. Сравнение групп признаков\n\n"
       "Для трёх профилей применяется Logistic Regression с одинаковой сеткой и training-CV. "
       "Добавление семестровых результатов проверяет ценность дополнительной информации. "
       "Это историческая ablation; доступность всех статусов в конкретную дату не установлена.")
    code("display(pd.read_csv(ROOT / 'artifacts/tables/feature_ablation.csv').round(4))\n"
         "display(Image(filename=str(ROOT / 'artifacts/figures/feature_ablation.png')))")
    md("## 11. Ошибки, интерпретация и ограничения\n\n"
       "Confusion matrix показывает, какие исходы путаются. Ошибка Dropout → Graduate "
       "может означать пропущенную потребность в поддержке; обратная ошибка — лишнее предупреждение. "
       "Enrolled остаётся отдельным исходом. Групповые метрики описательные и зависят от размера группы.")
    code("display(Image(filename=str(ROOT / 'artifacts/figures/selected_confusion_matrix.png')))\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/test_error_groups.csv').round(4))\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/selected_test_errors.csv').head(10))\n"
         "display(pd.read_csv(ROOT / 'artifacts/tables/validation_permutation_importance.csv').head(10).round(4))")
    md("Permutation importance вычислена на validation. Коррелированные исходные и производные признаки "
       "затрудняют индивидуальную интерпретацию; это не причинные эффекты. "
       "Один вуз, исторические наборы студентов, отсутствие индивидуальных дат отчисления и "
       "неполная доступность временных статусов ограничивают переносимость результата. "
       "Предсказания не являются основанием для автоматического отчисления.")
    matrix = pd.read_csv(ROOT / "artifacts/tables/selected_confusion_matrix.csv", index_col=0)
    enrolled_correct = int(matrix.loc["Enrolled", "Enrolled"])
    enrolled_total = int(matrix.loc["Enrolled"].sum())
    dropout_total = int(matrix.loc["Dropout"].sum())
    md(f"ANN ошиблась на {result['error_analysis']['incorrect']} из {result['error_analysis']['test_rows']} "
       f"test-наблюдений. Для Enrolled верно распознаны {enrolled_correct}/{enrolled_total} "
       f"({enrolled_correct / enrolled_total:.1%}); этот класс неоднороден и пересекается с другими исходами. "
       f"Из {dropout_total} случаев Dropout {int(matrix.loc['Dropout', 'Graduate'])} названы Graduate, "
       f"а {int(matrix.loc['Dropout', 'Enrolled'])} — Enrolled. Разбор таких ошибок полезнее одной accuracy.")
    chosen = next(row for row in result["final_comparison"] if row["selected_on_validation"])
    baseline = next(row for row in result["final_comparison"] if row["model"] == "Dummy")
    md("## 12. Выводы и оставшаяся работа\n\n"
       f"{result['selected_model']} выбрана по validation Macro F1={chosen['validation_macro_f1']:.4f}; "
       f"test Macro F1={chosen['test_macro_f1']:.4f}, против {baseline['test_macro_f1']:.4f} у Dummy. "
       "Сравнение ANN, PCA, кластеров и профилей признаков основано на сохранённых экспериментах. "
       "Численная часть не заменяет обзор литературы, 2–3-страничный proposal, итоговый отчёт, "
       "презентацию и техническую подготовку каждого участника.")
    md("## Начальные источники\n\n"
       "- [UCI dataset, DOI 10.24432/C5MC89](https://doi.org/10.24432/C5MC89).\n"
       "- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). "
       "[Predicting Student Dropout and Academic Success](https://doi.org/10.3390/data7110146). *Data, 7*(11), 146.\n"
       "- [scikit-learn: preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html).\n"
       "- [scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).\n"
       "- [scikit-learn: MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).\n"
       "- [scikit-learn: clustering](https://scikit-learn.org/stable/modules/clustering.html).\n"
       "- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).\n\n"
       "Это начальные технические ссылки, а не завершённая библиография из 10+ преимущественно научных источников.")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (Students Dropout)", "language": "python", "name": "students-dropout"},
        "language_info": {"name": "python", "version": "3.14.8"},
    })
    destination = ROOT / "notebooks/students_dropout_project.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    print(f"Built {destination.relative_to(ROOT)} ({len(cells)} cells).")


if __name__ == "__main__":
    main()
