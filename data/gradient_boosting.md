# Gradient Boosting

## Definition

Gradient Boosting is an ensemble learning technique where trees are built **sequentially**, and each new tree corrects the errors of previous trees.

---

## Working

1. Train first tree.
2. Compute residual errors.
3. Train next tree on residuals.
4. Repeat until errors are minimized.

---

## Key Idea

Each tree learns from previous mistakes.

---

## Popular Algorithms

- Gradient Boosting Machine (GBM)
- XGBoost
- LightGBM
- CatBoost

---

## Advantages

- Very high accuracy
- Handles nonlinear relationships
- Excellent predictive performance

---

## Disadvantages

- Slow training
- Sensitive to hyperparameters
- Can overfit

---

## Important Hyperparameters

- Learning Rate
- Number of Trees
- Maximum Depth
- Subsample
- Minimum Child Weight

---

## Interview Questions

**Q1. Difference between Bagging and Boosting?**

- Bagging builds trees independently.
- Boosting builds trees sequentially.

**Q2. Why use Learning Rate?**

- Controls contribution of each tree.

**Q3. Why does boosting reduce bias?**

- Each tree corrects previous errors.

---

## Applications

- Kaggle competitions
- Credit risk
- Recommendation systems
- Customer churn prediction
