# Home Credit Risk Prediction

## 1. Project Overview

This project aims to develop a machine learning model to predict the probability of loan default for Home Credit applicants.

Home Credit provides loans to individuals with limited or no credit history. The main challenge is to assess creditworthiness using alternative data sources and identify applicants who are likely to default.

This project formulates the problem as a **binary classification task**, where:

- `target = 1`: The applicant defaults.
- `target = 0`: The applicant does not default.

The model generates default probabilities that can be used to rank applicants by credit risk and convert predictions into credit scores.

## 2. Project Objectives

- Explore and understand the Home Credit dataset.
- Perform data cleaning and preprocessing.
- Develop meaningful financial and credit-related features.
- Reduce feature dimensionality using statistical methods.
- Train and compare multiple machine learning models.
- Optimize model hyperparameters.
- Evaluate predictive performance and temporal stability.
- Convert predicted probabilities into credit scores and risk grades.

## 3. Dataset

**Source:** [Home Credit - Credit Risk Model Stability (Kaggle)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)

The project uses the following training datasets:

| Dataset | Description |
|---|---|
| `train_base.csv` | Main application-level data |
| `train_static_0_0.csv` | Static applicant information |
| `train_static_0_1.csv` | Additional static applicant information |
| `train_static_cb_0.csv` | Credit bureau static information |

The datasets are merged using `case_id` as the primary identifier.

**Target variable:** `target`

**Time-related variable:** `WEEK_NUM`, used to preserve the chronological order of observations.

## 4. Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Polars | Data manipulation and processing |
| Pandas | Data handling and compatibility |
| NumPy | Numerical computations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Preprocessing, modeling and evaluation |
| LightGBM | Gradient boosting model |
| Optuna | Hyperparameter optimization |
| KaggleHub | Dataset downloading |
| Jupyter Notebook | Experimental environment |

## 5. Project Workflow

The project follows the following machine learning pipeline:

### 5.1. Data Loading and Understanding

- Download datasets from Kaggle using KaggleHub.
- Merge multiple data sources using `case_id`.
- Examine dataset dimensions, data types and descriptive statistics.
- Analyze target distribution and missing values.
- Investigate temporal characteristics using `WEEK_NUM`.

### 5.2. Data Splitting

The dataset is divided chronologically based on `WEEK_NUM`:

| Dataset | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

This approach ensures that the model is evaluated on future observations rather than randomly shuffled data.

### 5.3. Data Cleaning

The preprocessing stage includes:

- Converting columns into appropriate data types.
- Removing features with more than 75% missing values.
- Removing constant features.
- Handling missing values through model-specific imputation pipelines.

### 5.4. Feature Engineering

New features are constructed to capture applicants' financial conditions and credit behavior.

Major feature groups include:

- **Missingness Features:** Number and ratio of missing values.
- **Age Features:** Applicant age derived from available birth dates.
- **Affordability Features:** Credit-to-income, debt-to-income and annuity-to-income ratios.
- **Credit Utilization:** Outstanding debt, overdue debt and installment ratios.
- **Payment Behavior:** Late and early payment rates.
- **DPD Risk Features:** Maximum and average days past due.
- **Application Behavior:** Historical loan rejection indicators.
- **Credit History:** Credit contract and installment activity.
- **Cash Flow Features:** Payment and deposit balances relative to income.
- **Date Features:** Recency and calendar-related variables.

### 5.5. Feature Selection and Reduction

The project applies multiple techniques to identify informative features:

1. **Information Value (IV):** Evaluate the predictive power of individual features using Weight of Evidence (WoE).
2. **Gini-based Ranking:** Rank numerical features according to their individual discrimination power.
3. **Correlation-based Reduction:** Remove highly correlated numerical features using an absolute correlation threshold of 0.85.

### 5.6. Model Development

Four machine learning algorithms are implemented and compared:

| Model | Approach | Main Characteristics |
|---|---|---|
| Logistic Regression | Linear classification | Interpretable baseline |
| Decision Tree | Single decision tree | Rule-based predictions |
| Random Forest | Bagging | Reduces variance through ensemble learning |
| LightGBM | Gradient boosting | Efficient learning for structured data |

Model-specific preprocessing pipelines are applied:

- Logistic Regression: Median imputation, standardization and one-hot encoding.
- Decision Tree and Random Forest: Median imputation and ordinal encoding.
- LightGBM: Native handling of numerical and categorical features through the implemented model pipeline.

### 5.7. Hyperparameter Tuning

Hyperparameter optimization is performed for:

- **Logistic Regression:** RandomizedSearchCV with TimeSeriesSplit.
- **LightGBM:** Optuna with a TPE sampler and 20 optimization trials.

The optimization process uses ROC-AUC as its scoring metric.

### 5.8. Model Evaluation

Models are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures classification discrimination |
| Gini | Measures ranking power |
| Gini Stability | Evaluates predictive performance stability over time |
| Precision | Measures the proportion of predicted defaults that are actual defaults |
| Recall | Measures the proportion of actual defaults correctly identified |
| F1-Score | Balances precision and recall |

The classification threshold is selected using the validation set to maximize F1-score.

### 5.9. Credit Score Development

The best LightGBM model is used to generate credit risk predictions.

The scoring system includes:

- Probability of Default (PD) calibration.
- Conversion of calibrated PD into credit scores.
- Risk grade assignment.
- Default rate analysis across score groups.
- Credit score distribution analysis.
- Feature importance analysis using LightGBM Gain.

The scorecard uses the following parameters:

| Parameter | Value |
|---|---:|
| Base Points | 600 |
| Base Odds | 50 |
| Points to Double the Odds (PDO) | 20 |
| Minimum Score | 300 |
| Maximum Score | 850 |

Risk grades are assigned as follows:

| Credit Score | Risk Grade |
|---|---|
| ≥ 660 | A - Excellent |
| 620–659 | B - Good |
| 580–619 | C - Average |
| 540–579 | D - Bad |
| < 540 | E - High Risk |

**Note:** This project is developed for educational and research purposes. The resulting models and credit scores are experimental and should not be used as standalone tools for real-world lending decisions.
