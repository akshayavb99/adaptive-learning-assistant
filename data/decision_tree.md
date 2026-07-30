# Decision Tree

## Definition

A Decision Tree is a supervised learning algorithm that splits data into branches based on feature values.

---

## Structure

- Root Node
- Internal Node
- Leaf Node

---

## Splitting Criteria

Classification:

- Gini Index
- Entropy (Information Gain)

Regression:

- Variance Reduction

---

## Gini Index

\[
1-\sum p_i^2
\]

Lower Gini indicates purer nodes.

---

## Entropy

\[
-\sum p_i\log_2(p_i)
\]

---

## Advantages

- Easy to visualize
- Handles numerical and categorical data
- No feature scaling required

---

## Disadvantages

- Overfitting
- High variance
- Sensitive to data changes

---

## Pruning

- Pre-pruning
- Post-pruning

Purpose:
Reduce overfitting.

---

## Interview Questions

**Q1. Difference between Gini and Entropy?**

- Gini is faster.
- Entropy is more computationally expensive.

**Q2. Why do Decision Trees overfit?**

- They continue splitting until leaves become pure.

**Q3. How is overfitting prevented?**

- Pruning
- Limiting depth
- Minimum samples per leaf

---

## Applications

- Loan approval
- Medical diagnosis
- Fraud detection
- Risk assessment
