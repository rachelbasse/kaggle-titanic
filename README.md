# Lessons?

Frustratingly, the very first 'let's just test this out' logistic-regression model trained on minimally-cleaned data was the most accurate. However, it scored only slightly better than a decision tree stump split only on a binary feature representing an adult male (79% vs. 76% acc.).

AdaBoost performed noticieably worse as its weak estimators were strengthened and noticeably better as individual estimators were given less influence. The effect of the number of estimators fluctuated.

Tuning minimum sample sizes for splits in decision trees was helpful, presumably by reducing overfitting.

Feature importance varied wildly for AdaBoost and Random forests but was very consistent for single decision trees, where measures of sex, age, and class dominated.

A few derived features were helpful in tree-based classifiers:

- honorifics ('mr', 'mrs', 'miss', 'master', 'lass') representing mainly sex and age, extracted from the `name` and, for 'lass', `age` fields.
- a 4-valued ordinal combination of passenger-class and fare.
- a dangerously specific binary category representing females more likely to die.

Most derived features weren't helpful:

- Two- to six-valued categorical name origins (e.g., English, Western Europe, Eastern Europe/Western Asia) extracted from `name` field.
- ordinal combination of passenger-class and binary cabin (had a cabin number?).
- Combinations of `sex` with other fields.
- various measures of the amount of family also onboard.
- various features aiming to discriminate amongst adult males.

Over-sampling tended to slightly outperform class-weighting for decision trees; otherwise, they performed equally well. Class balancing was helpful for decision trees and for sex-separated models, which were extremely imbalanced.


# Top Performers

## Best

- Accuracy Score: 0.79425
- Preprocessing: Manual binning, missing data categorized as missing, minimum feature engineering, no feature selection, all features encoded as ordinal, no class balancing.
- Model: scikit-learn logistic regression, default hyperparams, all fields used except `ticket`.
- Commit: fee9ca7f39dabce3c4f13a7848a30e9c02f2f938

## Second

- Accuracy Score: 0.78947
- Preprocessing: Manual binning, imputed missing data, derived features, more feature selection, all features one-hot encoded, no class balancing.
- Model: scikit-learn AdaBoost classifier, tree-stump estimators, separate models for men/women, tuned hyperparams.
- Commit: 763157d2b410a3aa8ce6734ef10073e3cf1cf65a

## Third

- Accuracy Score: 0.78468
- Preprocessing: Manual binning, imputed missing data, derived features, more feature selection, categorical features one-hot encoded, classes balanced via oversampling minority.
- Model: scikit-learn decision tree classifier, tuned hyperparams.
- Commit: 891c6637e78b840514120f98398d83ea683585d6
