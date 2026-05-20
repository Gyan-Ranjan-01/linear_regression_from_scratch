# Linear Regression From Scratch

A raw NumPy implementation of Ordinary Least Squares (OLS) Linear Regression built from scratch, benchmarked directly against `scikit-learn`'s production model. This project demonstrates the underlying mathematics of gradient descent, cost functions, and matrix calculus without relying on high-level ML libraries for training.

## 🚀 Overview
The core objective of this project was to replicate `scikit-learn`'s `LinearRegression` model accuracy by implementing the mathematical foundations from first principles. 

* **Status:** Functional Jupyter Notebook (`.ipynb`)
* **Dataset:** Student Performance Dataset (predicting performance indexes based on metrics like study hours, sleep, and sample question papers).

---

## 📊 Core Mathematical Architecture

The model implements the standard linear formulation:

$$\hat{y} = Xw + b$$

Where:
* $X$ is the input feature matrix.
* $w$ is the weight vector (coefficients).
* $b$ is the bias (intercept).

### 1. Cost Function (Mean Squared Error)
To measure optimization performance, the model tracks Mean Squared Error (MSE):

$$J(w,b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 2. Gradient Descent Optimization
Partial derivatives are computed at each iteration to update parameters dynamically:

$$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (Xw + b - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (Xw + b - y)$$

---

## ⚡ Benchmarking: Custom vs. Scikit-Learn

The model was validated by splitting the `student_performance_dataset` into training/testing sets and running performance comparisons side-by-side. 

| Metric | Custom Implementation | Scikit-Learn (`LinearRegression`) |
| :--- | :--- | :--- |
| **R² Score** | `0.988` *(Update this)* | `0.988` *(Update this)* |
| **MSE** | `2.15` *(Update this)* | `2.15` *(Update this)* |
| **Weights ($w$)** | Match up to 4 decimal places | Match up to 4 decimal places |

> **Note:** Open your notebook and replace the placeholder numbers in the table above with your actual final R² scores and MSE values.

---

## 🛠️ Project Structure

```text
├── .gitignore                    # Prevents checkpoints and datasets from bloating repo
├── Linear_Regression_from_scrach.ipynb  # Core implementation notebook
└── student_performance_dataset.csv     # Training data
```

---

## How to Run

1. Clone the repository:
git clone [https://github.com/Gyan-Ranjan-01/linear_regression_from_scratch.git](https://github.com/Gyan-Ranjan-01/linear_regression_from_scratch.git)
cd linear_regression_from_scratch

2. Install dependencies:
pip install numpy pandas matplotlib scikit-learn jupyter

3. Launch the notebook:
jupyter notebook Linear_Regression_from_scrach.ipynb