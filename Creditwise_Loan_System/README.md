# Credit Wise Loan System – Intelligent Loan Approval Prediction System

## Overview

Credit Wise Loan System is an end-to-end Machine Learning project designed to predict whether a customer's loan application should be approved based on demographic, financial, and credit-related information.

The project implements a complete supervised machine learning pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and performance evaluation. Since the target variable contains two possible outcomes (Approved / Rejected), the problem is treated as a Binary Classification task.

Multiple machine learning algorithms were evaluated, including Logistic Regression, K-Nearest Neighbors (KNN), and Naive Bayes. Logistic Regression delivered the best overall performance and was selected as the final model.

---

## Problem Statement

Financial institutions receive thousands of loan applications every day. Manual assessment can be time-consuming and inconsistent.

The objective of this project is to build a machine learning system capable of predicting loan approval status based on applicant information, helping streamline the decision-making process.

---

## Dataset

Dataset Name: loan_approval_data.csv

Source: Kaggle

The dataset contains applicant-related information such as:

- Gender
- Marital Status
- Dependents
- Education
- Self Employment Status
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

and several additional attributes used for prediction.

Target Variable:

- Loan Status
  - Approved
  - Rejected

---

## Project Workflow

```text
Data Collection
       ↓
Data Understanding
       ↓
Exploratory Data Analysis
       ↓
Missing Value Treatment
       ↓
Outlier Analysis
       ↓
Feature Encoding
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Loan Approval Prediction
```

---

## Exploratory Data Analysis (EDA)

Several exploratory analysis techniques were performed to understand the dataset and identify patterns.

### Class Distribution Analysis

Since the problem is a classification task, the distribution of approval and rejection classes was analyzed to identify any class imbalance.

### Category-Wise Analysis

Different categorical features were analyzed against the target variable to understand their impact on loan approval.

### Box Plot Analysis

Box plots were used to:

- Identify outliers
- Understand feature distributions
- Compare numerical features across categories

### Correlation Analysis

A correlation heatmap was generated to identify strong linear relationships between numerical variables.

---

## Data Preprocessing

### Missing Value Handling

Both numerical and categorical features contained missing values.

Numerical Features:

Examples:

- Age
- Income
- Loan Amount

Missing values were handled using statistical measures such as:

- Mean
- Median

Categorical Features:

Examples:

- Gender
- Education
- Property Area

Missing values were handled using:

- Mode (Most Frequent Value)

The `select_dtypes()` method was used to separate numerical and categorical columns for preprocessing.

---

## Outlier Treatment

Outliers were carefully analyzed before removal.

### Meaningful Outliers

Some outliers represent genuine business scenarios and were retained.

Examples:

- High-income applicants
- Large loan amounts

### Non-Meaningful Outliers

Incorrect or impossible values were removed.

Examples:

- Age = 150
- Dependents = -1

Such values do not represent realistic customer information and can negatively impact model performance.

---

## Feature Encoding

Machine learning algorithms require numerical input. Therefore categorical variables were converted into numerical representations.

### Label Encoding

Used for ordinal or binary categories.

Example:

```python
Yes → 1
No → 0
```

### One-Hot Encoding

Used for nominal categories.

Example:

```python
Property_Area_Urban
Property_Area_Rural
Property_Area_SemiUrban
```

Techniques Used:

- Pandas `get_dummies()`
- Scikit-Learn Encoding Utilities

---

## Feature Engineering

Feature engineering was performed to improve model learning.

Techniques included:

- Creating transformed features
- Generating polynomial features
- Creating squared feature variations where relevant
- Evaluating feature importance and impact on model performance

The objective was to capture additional relationships not directly visible in the raw dataset.

---

## Machine Learning Models

The following classification algorithms were trained and evaluated:

### Logistic Regression

A linear classification algorithm used as the primary baseline model.

### K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts classes based on neighboring observations.

### Naive Bayes

A probabilistic classification algorithm based on Bayes' Theorem.

---

## Model Evaluation

Models were evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics provided a balanced understanding of model performance beyond simple accuracy.

---

## Final Model

After comparing all models, Logistic Regression achieved the best overall performance and was selected as the final model.

Reasons for selection:

- Better generalization
- Consistent classification performance
- Strong Precision, Recall, and F1 Score
- Simpler and more interpretable model

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

### Development Environment

- Jupyter Notebook

---

## Project Structure

```text
CreditWise Loan/
│
├── data/
│   └── loan_approval_data.csv
│
├── notebooks/
│   └── credit_wise.ipynb
│
├── models/In the code directory
│
├── images/
│
├── README.md
```

---

## Key Learnings

Through this project, I gained hands-on experience in:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Missing Value Treatment
- Outlier Detection
- Feature Encoding
- Feature Engineering
- Binary Classification
- Model Selection
- Performance Evaluation
- End-to-End Machine Learning Pipelines

---

## Future Improvements

Potential improvements include:

- Hyperparameter Tuning
- Cross Validation
- Ensemble Methods
- XGBoost Implementation
- Model Deployment
- Real-Time Prediction Interface

---
