# Linear Regression

## Definition

Linear Regression models the relationship between one or more independent variables and a dependent variable.

---

# Equation

Simple Linear Regression

$$
y=\beta_0+\beta_1x+\epsilon
$$

Multiple Linear Regression

$$
y=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n+\epsilon
$$

---

# Cost Function

Mean Squared Error

$$
J(\theta)=\frac1{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2
$$

---

# Gradient Descent

Parameter update:

$$
\theta_j=\theta_j-\alpha\frac{\partial J}{\partial\theta_j}
$$

where

- $\alpha$ = Learning rate

---

# Assumptions

- Linearity
- Independence
- Homoscedasticity
- Normality
- No multicollinearity

---

# Advantages

- Simple
- Fast
- Interpretable

---

# Disadvantages

- Sensitive to outliers
- Assumes linearity
- Cannot model complex patterns

---

# Evaluation Metrics

## MAE

$$
MAE=\frac1n\sum |y-\hat y|
$$

---

## MSE

$$
MSE=\frac1n\sum(y-\hat y)^2
$$

---

## RMSE

$$
RMSE=\sqrt{MSE}
$$

---

## R² Score

$$
R^2=1-\frac{SS_{res}}{SS_{tot}}
$$

---

# Applications

- Sales forecasting
- House price prediction
- Stock trend estimation
- Demand forecasting

---

# Exam Points

- Continuous output
- Parametric algorithm
- Uses least squares estimation
