# Random Forest

## Definition

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees.

---

# Working

1. Draw bootstrap samples.
2. Train multiple Decision Trees.
3. Randomly select subsets of features at each split.
4. Aggregate predictions.

---

# Bootstrap Sampling

Each tree is trained on a random sample with replacement.

---

# Voting

Classification

$$
\hat y=\operatorname{mode}(T_1,T_2,\ldots,T_n)
$$

Regression

$$
\hat y=\frac1n\sum_{i=1}^{n}T_i
$$

---

# Why It Works

Randomness reduces variance.

Overall prediction becomes more robust.

---

# Hyperparameters

- Number of trees
- Maximum depth
- Minimum samples split
- Maximum features

---

# Advantages

- High accuracy
- Reduces overfitting
- Handles missing values
- Robust to noise

---

# Disadvantages

- Slow prediction
- Large memory usage
- Less interpretable

---

# Applications

- Fraud detection
- Medical diagnosis
- Recommendation systems
- Remote sensing
- Finance

---

# Comparison with Decision Tree

| Decision Tree   | Random Forest    |
| --------------- | ---------------- |
| Single tree     | Multiple trees   |
| High variance   | Lower variance   |
| Overfits easily | Less overfitting |
| Faster          | Slower           |

---

# Exam Points

- Ensemble method
- Bagging algorithm
- Non-parametric
- Better generalisation than Decision Trees

```

```
