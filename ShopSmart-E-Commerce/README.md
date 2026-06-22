# ShopSmart — Online Shopper Purchase Intention Prediction

### Assignment 4 | Supervised Machine Learning | Decision Tree Classifier

---

## Problem Statement

**ShopSmart** wants to predict which website visitors are likely to make a purchase based on their session behaviour.

- **Dataset:** 12,330 individual user sessions (1 year)
- **Target:** `Revenue` — whether the visitor purchased (binary: True/False)
- **Challenge:** Dataset is **imbalanced** (~84% No Purchase, ~15% Purchase)
- **Model Required:** Decision Tree Classifier with Pruning
- **Benchmark:** F1-Score ≥ **0.55**

---

## Project Structure

```
shopSmart-ml/
│
├── solution.ipynb                       # Full solution (EDA → Model → Evaluation)
├── shop_smart_ecommerce.csv                      # Dataset (replace with actual UCI dataset)
├── README.md                         # This file
│
├── eda_target_distribution.png       # Class imbalance visualisation
├── eda_num_distributions.png         # Numerical feature histograms
├── eda_correlation_heatmap.png       # Feature correlation matrix
├── eda_categorical_vs_revenue.png    # Categorical features vs purchase rate
├── eda_pagevalues_vs_revenue.png     # PageValues vs Revenue
│
├── pruning_alpha_vs_f1.png           # CCP Alpha selection plot
├── feature_importance.png            # Top predictive features
├── decision_tree_visualisation.png   # Tree structure (first 3 levels)
├── eval_pruned_decision_tree.png     # Confusion Matrix + ROC Curve
└── final_comparison.png              # Baseline vs Pruned vs Benchmark
```

---

## Dataset

**Source:** [UCI ML Repository — Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

Download `online_shoppers_intention.csv` and place it in the project root as `shop_smart_ecommerce.csv`.

| Feature                 | Type                  | Description                                     |
| ----------------------- | --------------------- | ----------------------------------------------- |
| Administrative          | Numerical             | Pages visited on account/login section          |
| Administrative_Duration | Numerical             | Time spent on administrative pages (seconds)    |
| Informational           | Numerical             | Informational pages visited                     |
| Informational_Duration  | Numerical             | Time on informational pages (seconds)           |
| ProductRelated          | Numerical             | Product pages visited                           |
| ProductRelated_Duration | Numerical             | Time on product pages (seconds)                 |
| BounceRates             | Numerical             | % visitors leaving after 1 page                 |
| ExitRates               | Numerical             | % exits from a given page                       |
| PageValues              | Numerical             | Avg value of pages visited before a transaction |
| SpecialDay              | Numerical             | Closeness to a special day (0–1)                |
| Month                   | Categorical           | Month of visit                                  |
| OperatingSystems        | Categorical (encoded) | OS used by visitor                              |
| Browser                 | Categorical (encoded) | Browser used                                    |
| Region                  | Categorical (encoded) | Geographic region                               |
| TrafficType             | Categorical (encoded) | Traffic source                                  |
| VisitorType             | Categorical           | Returning / New / Other                         |
| Weekend                 | Boolean               | Was visit on a weekend?                         |
| **Revenue**             | **Target**            | **Did visitor purchase? (True/False)**          |

---

## Setup & Installation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Run the Solution

```bash
python solution.ipynb
```

> **Note:** Grid search can take 5–10 minutes depending on your machine. To run faster, reduce `param_grid` range in `tune_decision_tree()`.

---

## Methodology

### Step 1 — Exploratory Data Analysis (EDA)

**What we check:**

- Shape, data types, null values, duplicates
- Target class distribution → confirms severe imbalance (~15% positive)
- Numerical feature distributions → identifies right-skewed columns
- Correlation heatmap → finds multicollinearity (BounceRates ↔ ExitRates)
- Categorical features vs purchase rate → Month, VisitorType patterns

**Key finding:** `PageValues` is the strongest predictor of purchase intent.

---

### Step 2 — Feature Preprocessing

| Transformation      | Applied To               | Reason                                  |
| ------------------- | ------------------------ | --------------------------------------- |
| Drop duplicates     | Entire dataset           | Prevent data leakage / bias             |
| Ordinal encoding    | `Month`                  | Calendar order matters                  |
| Label encoding      | `VisitorType`            | 3 nominal categories                    |
| Boolean → int       | `Weekend`, `Revenue`     | Model-friendly format                   |
| `log1p()` transform | 7 skewed numeric columns | Reduces right-skew, stabilises variance |

---

### Step 3 — Train/Test Split

```
80% Training | 20% Test | stratify=y (preserves class ratio in both sets)
```

---

### Step 4 — Handling Class Imbalance with SMOTE

**Problem:** With ~15% positive class, a naive model achieves ~84% accuracy by always predicting "No Purchase" — useless for business.

**Solution:** SMOTE (Synthetic Minority Over-sampling Technique)

- Creates synthetic samples of the minority class (Purchase=True)
- Applied **only on training data** — never on test data (prevents data leakage)
- Result: Balanced 50/50 training set

---

### Step 5 — Decision Tree Classifier

**Baseline (Unpruned):**

- No depth limit → tree memorises training data (overfitting)
- Good training F1, lower test F1

**Pruning Strategy:**

**1. Post-Pruning via Cost-Complexity Pruning (`ccp_alpha`)**

- sklearn computes a "pruning path" of alpha values
- Higher alpha → more nodes pruned → simpler tree
- Best alpha selected using 5-fold cross-validation on F1

**2. Pre-Pruning via Hyperparameter Constraints**

```python
max_depth          # limits tree depth
min_samples_split  # min samples to allow a split
min_samples_leaf   # min samples required at each leaf
class_weight       # handles residual imbalance
```

**3. GridSearchCV** — exhaustive search over all combinations, 5-fold CV, F1 scoring

---

### Step 6 — Evaluation

**Primary metric: F1-Score**

> F1 = 2 × (Precision × Recall) / (Precision + Recall)

F1 is used (not Accuracy) because the dataset is imbalanced.
A model predicting "No Purchase" always gets ~84% accuracy but 0% F1 on the purchase class — useless.

**Additional metrics reported:**

- Precision, Recall (per class)
- ROC-AUC Score
- Confusion Matrix

---

## Results

| Model                 | F1-Score  | Notes                  |
| --------------------- | --------- | ---------------------- |
| Benchmark             | 0.55      | Assignment target      |
| Baseline (Unpruned)   | ~0.75     | Overfits training data |
| **Pruned DT (Final)** | **~0.86** | ✅ Above benchmark     |

> Actual results will vary slightly with the real UCI dataset.

---

## Why Decision Tree?

| Property            | Benefit                                                                             |
| ------------------- | ----------------------------------------------------------------------------------- |
| Interpretable       | Easy to explain to stakeholders ("If PageValues > X AND BounceRate < Y → Purchase") |
| No scaling needed   | Decision Trees don't need StandardScaler                                            |
| Handles mixed types | Numerical + categorical features together                                           |
| Pruning available   | Controls overfitting directly                                                       |

---

## Key Insights (Business Value)

1. **PageValues** is the top predictor — pages that lead to transactions are strong purchase signals. Optimise landing page content.
2. **BounceRates** are high among non-buyers — reducing bounce improves conversion.
3. **Returning visitors** convert at a higher rate than new visitors.
4. **November & October** sessions have the highest purchase rates (pre-holiday season).

---

## Author

**Jay Patel**
