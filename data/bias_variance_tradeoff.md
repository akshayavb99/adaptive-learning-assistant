# Bias-Variance Tradeoff

## Definition

Bias-Variance Tradeoff describes the balance between **underfitting** and **overfitting**.

---

## Bias

High Bias:

- Model is too simple.
- Misses important patterns.
- Causes underfitting.

Examples:

- Linear Regression on nonlinear data.

---

## Variance

High Variance:

- Model memorizes training data.
- Performs poorly on unseen data.
- Causes overfitting.

Examples:

- Deep Decision Trees.

---

## Total Error

\[
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
\]

---

## Underfitting

Characteristics:

- High training error
- High testing error

Solutions:

- Increase model complexity
- Add features
- Reduce regularization

---

## Overfitting

Characteristics:

- Low training error
- High testing error

Solutions:

- Cross-validation
- Regularization
- Pruning
- More data
- Ensemble methods

---

## Bias vs Variance

| High Bias           | High Variance       |
| ------------------- | ------------------- |
| Underfitting        | Overfitting         |
| Simple model        | Complex model       |
| High training error | Low training error  |
| Poor generalization | Poor generalization |

---

## Interview Questions

**Q1. What causes overfitting?**

- Excessively complex models.
- Too many features.
- Small datasets.

**Q2. How do Random Forests reduce variance?**

- By averaging predictions from multiple trees.

**Q3. How does Boosting reduce bias?**

- By sequentially correcting previous errors.

**Q4. Which models usually have high bias?**

- Linear Regression
- Logistic Regression

**Q5. Which models usually have high variance?**

- Deep Decision Trees

---

## Key Takeaway

- High Bias → Underfitting
- High Variance → Overfitting
- The objective is to achieve the right balance for optimal generalization.
