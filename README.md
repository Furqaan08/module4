# Module 4: Scikit-Learn ML Pipeline — Coding Assignments

Two coding assignments completed as part of Module 4 covering
Scikit-Learn pipelines, preprocessing, hyperparameter tuning,
model evaluation, and rule-based benchmarking.

---

## Section D: Coding Exercises

### Exercise 1 — GridSearchCV Pipeline
Built a pipeline combining StandardScaler and LogisticRegression,
then used GridSearchCV to test different values of the C parameter
across 5 cross-validation folds. Printed the best parameter found
and the best cross-validation accuracy score.

### Exercise 2 — Scaler Comparison
Built two separate pipelines — one with StandardScaler and one
with QuantileTransformer — both paired with LogisticRegression.
Compared accuracy of both on the same dataset and printed
which scaler performed better.

Dataset used: Breast Cancer Wisconsin (built-in Scikit-Learn)
Output: Both exercises are in `section_d.ipynb`

---

## Section E: Main Assignment — Complete ML Pipeline

**Dataset:** Breast Cancer Wisconsin Dataset
**Source:** sklearn.datasets.load_breast_cancer
**Problem:** Binary classification — predicting whether a tumor
is malignant or benign based on 30 numerical features.

### Tasks Completed
- Dataset loaded, explored, and documented
- Data split using 80:20 ratio
- Pipeline built with StandardScaler + LogisticRegression
- Hyperparameter tuning done using GridSearchCV testing
  multiple values of C and two solvers
- Model evaluated using Accuracy, Precision, Recall,
  F1-Score, and Confusion Matrix
- Rule-based benchmark built using the worst radius feature —
  a medically relevant threshold was identified from training
  data and applied on test data
- Final comparison table printed between ML model
  and rule-based benchmark

Output: Full project is in `section_e_pipeline.ipynb`

---
