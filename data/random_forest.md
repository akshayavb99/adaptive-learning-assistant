# Random Forest

## Definition

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees using **Bagging**.

---

## Working

1. Draw bootstrap samples.
2. Train one tree on each sample.
3. Select random subset of features at each split.
4. Aggregate predictions.

Classification:

- Majority Voting

Regression:

- Average Prediction

---

## Why Random Feature Selection?

Reduces correlation among trees.

---

## Advantages

- High accuracy
- Reduces overfitting
- Handles missing values
- Robust to noise

---

## Disadvantages

- Slow prediction
- Less interpretable
- Large memory usage

---

## Important Hyperparameters

- n_estimators
- max_depth
- max_features
- min_samples_split
- min_samples_leaf

---

## Interview Questions

**Q1. Why better than Decision Tree?**

- Lower variance.
- Less overfitting.

**Q2. What is Bagging?**

- Bootstrap Aggregating.

**Q3. Why bootstrap sampling?**

- Produces diverse trees.

---

## Applications

- Credit scoring
- Disease prediction
- Fraud detection
- Customer segmentation
