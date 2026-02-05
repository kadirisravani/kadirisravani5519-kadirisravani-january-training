import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/datasets.csv")

print(df.info())
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill numerical missing values with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after treatment:")
print(df.isnull().sum())

# Convert TotalCharges to numeric (common issue in churn dataset)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Refill after conversion
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.clip(df[col], lower, upper)

    print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# CustomerID is not useful for modeling
df.drop(columns=['customerID'], inplace=True)

df.to_csv("data/cleaned_dataset.csv", index=False)

import numpy as np

# Log transformation for skewed columns
skewed_cols = df[num_cols].skew().sort_values(ascending=False)
high_skew = skewed_cols[skewed_cols > 1].index

for col in high_skew:
    df[col] = np.log1p(df[col])