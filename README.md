# 🏢 Real Estate Valuation Machine Learning Pipeline (Tashkent Market)

> **An empirical data science case study executing multivariate linear regression and machine learning ensembles (K-NN, Random Forest, Gradient Boosting) to predict residential property valuations based on spatial, structural, and temporal features.**

---

## 📌 Project Overview & Objective

Predicting real estate valuation requires capturing both micro-structural property characteristics and macro-spatial geography. The goal of this project was to engineer a predictive pipeline using a dataset of individual property transactions across Tashkent, Uzbekistan, modeling the complex relationship between real estate prices and high-dimensional features.

### 📊 Dataset Core Attributes
* **Target Variable:** `Price_in_USD` (Continuous valuation)
* **Spatial/Geographical Features:** `Area` (Proximity to urban downtown), `Tashkent_city_Districts`, `Group` (Categorical district encodings), `Latitude`, and `Longitude`.
* **Structural Features:** `Size` (Square meters), `Rooms` (Total room count), and `Type` (Structural classification: New vs. Old construction).
* **Temporal Features:** `Date` (Year of construction completion).

---

## 🛠️ Methodological Architecture & Modeling Strategy

To solve this regression task, the project implements and benchmarks four distinct modeling paradigms, evaluating the trade-offs between statistical interpretability and algorithmic predictive power:

1. **Multivariate Linear Regression:** Established as the econometric baseline to compute explicit feature coefficients, evaluate statistical significance ($p$-values), and verify standard residual assumptions.
2. **K-Nearest Neighbors (K-NN) Regression:** A non-parametric instance-based model capturing localized spatial price clusters based on geographical and structural proximity.
3. **Random Forest Regression:** An ensemble bagging technique utilizing deeply grown parallel decision trees to capture non-linear feature interactions (e.g., how the impact of property `Size` changes across different `Districts`) while mitigating overfitting.
4. **Gradient Boosting Machine (GBM):** A sequential boosting architecture optimizing a loss function step-by-step to minimize residual errors and capture highly complex patterns in the housing market.

---

## 🔍 Core Pipeline Stages Documented

* **Exploratory Data Analysis (EDA):** Feature distribution profiling, mapping coordinate data (`Latitude`/`Longitude`) to isolate premium spatial zones, and evaluating feature correlation matrices.
* **Feature Engineering & Preprocessing:** Handling categorical variables through robust encoding schemas, managing scale variance for distance-based algorithms (K-NN), and treating potential outliers in property square footage.
* **Model Benchmarking & Evaluation:** Comparative evaluation utilizing robust regression metrics: Root Mean Squared Error (**RMSE**), Mean Absolute Error (**MAE**), and R-squared (**$R^2$**).

---

## 📈 Key Findings & Performance Metrics

The predictive models were evaluated using Mean Squared Error (MSE) and R-squared metrics to identify the most accurate valuation engine for the Tashkent housing market.

### 1. K-Nearest Neighbors (KNN) Spatial Optimization
Below is the hyperparameter tuning visualization for the KNN Regression model, plotting the relationship between the number of neighbors ($K$) and the model's prediction error:

<img width="1152" height="712" alt="Knn" src="https://github.com/user-attachments/assets/fd318131-d6bc-4fd0-9447-efc7dd2c853b" />

* **Analysis:** The chart demonstrates a classic bias-variance trade-off curve. As $K$ increases from 1, the model's error drops sharply as it filters out localized data noise. It reaches an optimal peak performance point before the error begins rising again, which occurs because a high $K$ value forces the model to look too far outside a property's immediate neighborhood, smoothing out real micro-market price differences.

### 2. Final Model Benchmarking
* **The Winning Model:** The **Gradient Boosting Machine (GBM)** model emerged as the definitive best-performing architecture, delivering the absolute lowest Mean Squared Error (MSE) across the test dataset. 
* **Why it Won:** While Linear Regression struggled with complex geometric boundaries and KNN was highly sensitive to scale, the sequential tree-boosting nature of GBM perfectly adapted to non-linear interactions between structural unit size and regional district pricing trends.
