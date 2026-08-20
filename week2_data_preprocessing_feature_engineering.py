# Week 2 - Data Preprocessing and Feature Engineering
# Machine Learning Internship Project
# Dataset: Titanic - Machine Learning from Disaster

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Place titanic.csv in the same folder as this Python file.
df = pd.read_csv("titanic.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 2. DATA QUALITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()
print(missing_values)

print("\nTotal missing cells:", df.isnull().sum().sum())


print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)


print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE PERCENTAGE")
print("=" * 60)

missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage[missing_percentage > 0])


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Extract title from passenger name
df["Title"] = (
    df["Name"]
    .str.extract(r",\s*([^.]*)\.", expand=False)
    .str.strip()
)

# Group rare titles
title_counts = df["Title"].value_counts()
rare_titles = title_counts[title_counts < 10].index

df["Title"] = df["Title"].replace(rare_titles, "Rare")

# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Whether passenger travelled alone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Approximate fare per person
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# Cabin availability indicator
df["HasCabin"] = df["Cabin"].notna().astype(int)

# Extract deck from cabin
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

print("New features created:")
print("- Title")
print("- FamilySize")
print("- IsAlone")
print("- FarePerPerson")
print("- HasCabin")
print("- Deck")


# ============================================================
# 5. SELECT FEATURES
# ============================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
    "HasCabin"
]

categorical_features = [
    "Sex",
    "Embarked",
    "Title",
    "Deck"
]

X = df[numeric_features + categorical_features]
y = df["Survived"]


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 7. PREPROCESSING PIPELINES
# ============================================================

# Numerical preprocessing:
# - Median imputation
# - Standard scaling

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing:
# - Most frequent value imputation
# - One-hot encoding

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine both preprocessing pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 8. FIT PREPROCESSING ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Training matrix shape:", X_train_processed.shape)
print("Testing matrix shape:", X_test_processed.shape)


# ============================================================
# 9. FEATURE CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)

correlation_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
    "HasCabin",
    "Survived"
]

correlation_matrix = df[correlation_features].corr()

print(
    correlation_matrix["Survived"]
    .sort_values(ascending=False)
)


# ============================================================
# 10. OUTLIER ANALYSIS USING IQR
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)


def count_iqr_outliers(series):
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (series < lower_bound) |
        (series > upper_bound)
    )

    return outliers.sum(), lower_bound, upper_bound


for column in [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "FarePerPerson"
]:

    count, lower, upper = count_iqr_outliers(df[column])

    print(
        f"{column}: {count} potential outliers "
        f"(lower={lower:.2f}, upper={upper:.2f})"
    )


# ============================================================
# 11. PCA EXPERIMENT
# ============================================================

print("\n" + "=" * 60)
print("PCA ANALYSIS")
print("=" * 60)

# Convert processed matrix to dense array if necessary
if hasattr(X_train_processed, "toarray"):
    X_train_dense = X_train_processed.toarray()
else:
    X_train_dense = X_train_processed

pca = PCA()

pca.fit(X_train_dense)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

components_for_90 = (
    np.argmax(cumulative_variance >= 0.90) + 1
)

print(
    "Components required for 90% variance:",
    components_for_90
)

pca_2 = PCA(n_components=2)

X_train_pca = pca_2.fit_transform(X_train_dense)

print(
    "Variance explained by 2 components:",
    pca_2.explained_variance_ratio_.sum()
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 60)

print("Original dataset shape:", df.shape)

print("Original missing cells:",
      missing_values.sum())

print("Duplicate rows:",
      duplicate_count)

print("Engineered features:")
print(
    [
        "Title",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
        "HasCabin",
        "Deck"
    ]
)

print("\nPreprocessing completed successfully.")

print("\nPipeline:")
print("Raw Data")
print("   ↓")
print("Train/Test Split")
print("   ↓")
print("Missing Value Treatment")
print("   ↓")
print("Feature Engineering")
print("   ↓")
print("Categorical Encoding")
print("   ↓")
print("Feature Scaling")
print("   ↓")
print("PCA / Feature Selection")
print("   ↓")
print("ML-Ready Data")
