
# Week 2 – Data Preprocessing and Feature Engineering

## Machine Learning Internship Project

This project demonstrates a complete data preprocessing and feature engineering workflow using the **Titanic – Machine Learning from Disaster** dataset.

The objective is to transform raw real-world data into a clean, consistent, and machine-learning-ready representation while applying appropriate techniques for missing values, categorical variables, outliers, feature engineering, feature selection, scaling, dimensionality reduction, and data leakage prevention.

---

## Project Objectives

- Understand and inspect a real-world dataset
- Identify missing values and duplicate records
- Perform data quality assessment
- Handle missing values using appropriate strategies
- Detect and analyze potential outliers
- Engineer meaningful features
- Encode categorical variables
- Scale numerical features
- Analyze feature relationships
- Apply PCA as an optional dimensionality-reduction technique
- Prevent data leakage during preprocessing
- Build a reproducible preprocessing pipeline using Scikit-learn

---

## Dataset

**Dataset:** Titanic – Machine Learning from Disaster

**Source:** Kaggle

The dataset contains passenger information such as:

- Passenger class
- Name
- Sex
- Age
- Number of siblings/spouses
- Number of parents/children
- Ticket
- Fare
- Cabin
- Port of embarkation
- Survival outcome

The dataset contains **891 passenger records and 12 original columns**.

---

## Data Quality Analysis

The initial analysis identified:

- **866 missing cells**
- **177 missing Age values**
- **687 missing Cabin values**
- **2 missing Embarked values**
- **0 exact duplicate rows**

These findings were used to determine the appropriate preprocessing strategy.

---

## Data Preprocessing

### Missing Value Treatment

Different strategies were selected based on feature type and missingness:

| Feature | Treatment |
|---|---|
| Age | Median imputation |
| Embarked | Most-frequent / mode imputation |
| Cabin | Missingness represented using engineered features |

For Cabin, instead of creating artificial cabin values, the project derives:

- `HasCabin`
- `Deck`

This preserves potentially useful information about cabin availability.

---

## Feature Engineering

The following features were created:

### FamilySize

```text
FamilySize = SibSp + Parch + 1
