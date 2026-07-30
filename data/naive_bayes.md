# Naive Bayes

## Definition

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem** with the assumption that all features are independent.

---

## Bayes Theorem

\[
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
\]

Where:

- P(A|B) = Posterior
- P(B|A) = Likelihood
- P(A) = Prior
- P(B) = Evidence

---

## Types

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

---

## Assumption

Features are conditionally independent.

---

## Advantages

- Very fast
- Works well on text data
- Handles high-dimensional datasets

---

## Disadvantages

- Independence assumption rarely holds
- Zero-frequency problem

---

## Solution for Zero Frequency

Laplace Smoothing

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

---

## Interview Questions

**Q1. Why is it called Naive?**

- Assumes independent features.

**Q2. Where does it perform well?**

- NLP
- Spam detection
- Document classification

**Q3. What is Laplace Smoothing?**

- Adds 1 to frequencies to avoid zero probability.

---

## Applications

- Email spam detection
- Sentiment analysis
- Document classification
- Recommendation systems
