# Treatment effectiveness evaluation
Survival analysis, Parametric tests (ANOVA, Tukey test), Nonparametric tests (Kruskal-Wallis, Mann-Whitney tests with Holm-Bonferroni adjustments), Analysis of repeated measures (ANOVA for repeated measures, Tukey test, Friedman test, Conover test)

...

В данном проекте требовалось оценить эффективность терапии.

Были предоставлены демографические данные пациентов (пол, возраст, диагноз, инвалидность, ИМТ, семейный статус, уровень образования и т.д.), данные оценки пациентов по различным шкалам в четырех контрольных точках по времени и данные о группе терапии и летальности. Для соблюдения конфиденциальности все данные обезличены и названия шкал не приводятся.

Пациенты были разбиты на три неравные группы терапии. Часть пациентов получала терапию 24 часа в сутки, часть по требованию.

Заказчик поставил задачу оценить эффективность терапии как в целом для всех участников, так и в группах терапии по отедельности. Группы терапии сравнивать между собой не сравнивались

Для этого дыл применен следующий дизайн оценки:

описательная статистика (летальность, группы терапии, демографические данных, шкал по визитам)
анализ выживаемости (в целом, с разделением по группам терапий)
для нормально распределенных данных (в целом, с разделением по группам терапий) - параметрические тесты изменения показателей шкал по визитам (ANOVA, Тьюки)
для ненормально распреледенных данных или категориальных переменных (в целом, с разделением по группам терапий) - непараметрические тесты изменения показателей шкал по визитам (Крускал-Уоллис, Манн-Уитни с поправкой Холма-Бонферрони)
анализ повторных измерений шкал по визитам для нормально распределенных данных - ANOVA для повторных изменений, тест Тьюки
анализ построных измерений шкал по визитам для ненормально распределенных данных - тест Фридмана, тест Коновера
Для оценки нормальности распределения использовала визульаный метод оценки qq-plot, гистограммы, а так же параметров эксцесса и асимметрии

В данном отчете представлен только расчет и визуализация. Описание результатов и выводы не представлены

...

In this project, it was necessary to evaluate the effectiveness of therapy.

Demographic data of patients were provided (sex, age, diagnosis, disability, BMI, family status, education level, etc.), as well as data on patients' assessment on various scales at four time points, therapy group and outcome(mortality). To maintain confidentiality, all data were anonymized, and the names of the scales were not provided.

Patients were divided into three unequal therapy groups. Some patients received therapy 24 hours a day, while others received it on demand.

The task given by the customer was to evaluate the effectiveness of therapy both overall for all participants and in therapy groups separately. Comparing therapy groups with each other was not feasible.

To achieve this, the following evaluation design was applied:

Descriptive statistics (outcome (mortality), therapy groups, demographic data, scales at visits)
Survival analysis (overall, in therapy groups separately)
Parametric tests for normally distributed data (overall, with division by therapy groups) - ANOVA, Tukey test
Nonparametric tests for non-normally distributed data or categorical variables (overall, with division by therapy groups) - Kruskal-Wallis, Mann-Whitney tests with Holm-Bonferroni adjustments
Analysis of repeated measurements of scale indicators for normally distributed data - ANOVA for repeated measures, Tukey test
Analysis of non-repeated measurements of scale indicators for non-normally distributed data - Friedman test, Conover test
To assess the normality of the distribution, a visual method of assessment was used, including QQ plots, histograms, and measures of kurtosis and skewness.

This report only includes calculation and visualization. Description of the results is not presented.
