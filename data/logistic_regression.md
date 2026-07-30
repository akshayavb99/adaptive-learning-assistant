# Logistic Regression

## Definition

Logistic Regression is a **classification algorithm** that predicts probabilities.

---

# Sigmoid Function

$$
\sigma(z)=\frac1{1+e^{-z}}
$$

where

$$
z=\beta_0+\beta_1x
$$

Output lies between

$$
0 \le P(y=1|x)\le1
$$

---

# Decision Rule

Predict

$$
1
$$

if

$$
P>0.5
$$

Otherwise predict

$$
0
$$

---

# Cost Function

Binary Cross Entropy

$$
J=-\frac1m\sum
\left[
y\log(\hat y)+(1-y)\log(1-\hat y)
\right]
$$

---

# Advantages

- Simple
- Fast
- Probabilistic output
- Easy interpretation

---

# Disadvantages

- Linear decision boundary
- Poor on highly nonlinear data

---

# Evaluation Metrics

- Accuracy
- Precision

$$
Precision=\frac{TP}{TP+FP}
$$

Recall

$$
Recall=\frac{TP}{TP+FN}
$$

F1-score

$$
F1=
\frac{2PR}{P+R}
$$

---

# Applications

- Spam detection
- Disease prediction
- Customer churn
- Fraud detection

---

# Exam Points

- Used for classification
- Uses sigmoid activation
- Parametric algorithm
