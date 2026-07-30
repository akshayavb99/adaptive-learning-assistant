# Inference and Prediction

## Introduction

Machine Learning serves two major purposes:

1. **Inference**
2. **Prediction**

---

# Inference

## Definition

Inference aims to understand **how input variables influence the output**.

The objective is explanation rather than prediction.

Example:

- Does age affect income?
- Does education impact salary?

---

## Goal

Estimate model parameters.

Example:

$$
y=\beta_0+\beta_1x+\epsilon
$$

Estimate:

$$
\beta_0,\ \beta_1
$$

---

## Statistical Inference

Includes

- Hypothesis testing
- Confidence intervals
- p-values
- Significance tests

---

# Prediction

## Definition

Prediction focuses on estimating future outcomes.

Example:

Predict house price.

---

## Goal

Minimize prediction error.

Common loss function:

$$
MSE=\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

---

# Comparison

| Inference                | Prediction              |
| ------------------------ | ----------------------- |
| Understand relationships | Predict future values   |
| Explain model            | Maximize accuracy       |
| Uses p-values            | Uses evaluation metrics |
| Statistical              | Machine Learning        |

---

# Metrics

Regression:

- MAE
- MSE
- RMSE
- R²

Classification:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

# Exam Points

- **Inference explains.**
- **Prediction forecasts.**
- Linear regression can be used for both.
