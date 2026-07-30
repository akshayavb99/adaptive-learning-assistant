# Parametric and Non-Parametric Methods

## Introduction

Machine Learning models can broadly be classified into **Parametric** and **Non-Parametric** methods depending on the assumptions they make about the underlying data distribution.

---

# Parametric Methods

## Definition

A **parametric model** assumes a specific functional form for the relationship between input and output.

The model is represented by a **fixed number of parameters**, regardless of the amount of training data.

Example:

$$
y = \beta_0 + \beta_1x
$$

Only two parameters need to be learned:

- $\beta_0$ (Intercept)
- $\beta_1$ (Slope)

---

## Characteristics

- Fixed number of parameters
- Faster training
- Simpler models
- Requires assumptions about data distribution
- Less flexible

---

## Examples

- Linear Regression
- Logistic Regression
- Naive Bayes
- Linear Discriminant Analysis

---

## Advantages

- Fast training
- Less data required
- Easy interpretation
- Lower computational cost

---

## Disadvantages

- Strong assumptions
- Can underfit complex data
- Limited flexibility

---

# Non-Parametric Methods

## Definition

A **non-parametric model** does **not assume a predefined functional form**.

Its complexity grows with the amount of training data.

---

## Characteristics

- Number of parameters increases with data
- Flexible
- Captures complex relationships
- Requires more data
- Higher computational cost

---

## Examples

- Decision Trees
- Random Forest
- K-Nearest Neighbours
- Support Vector Machines (RBF Kernel)

---

## Advantages

- High flexibility
- Few assumptions
- Better accuracy for complex datasets

---

## Disadvantages

- Slow training
- Memory intensive
- Higher risk of overfitting

---

# Comparison

| Feature          | Parametric | Non-Parametric |
| ---------------- | ---------- | -------------- |
| Assumptions      | Yes        | No             |
| Parameters       | Fixed      | Variable       |
| Flexibility      | Low        | High           |
| Training Speed   | Fast       | Slow           |
| Data Required    | Less       | More           |
| Overfitting Risk | Lower      | Higher         |

---

# Key Formula

Linear model:

$$
y = f(x;\theta)
$$

where

- $x$ = input
- $\theta$ = fixed parameters

---

# Exam Points

- **Parametric = Fixed parameters**
- **Non-parametric = Model complexity grows with data**
- Linear Regression is parametric.
- Decision Tree is non-parametric.
