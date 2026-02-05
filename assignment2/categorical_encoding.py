import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

df = pd.read_csv("data/datasets.csv")

cat_cols = df.select_dtypes(include="object").columns

one_hot_df = pd.get_dummies(df, columns=cat_cols)

label_df = df.copy()
le = LabelEncoder()

for col in cat_cols:
    label_df[col] = le.fit_transform(label_df[col])

    ordinal_df = df.copy()
oe = OrdinalEncoder()

ordinal_df[cat_cols] = oe.fit_transform(ordinal_df[cat_cols])

freq_df = df.copy()

for col in cat_cols:
    freq_df[col] = freq_df[col].map(freq_df[col].value_counts())

    target_df = df.copy()
te = TargetEncoder()

target_df[cat_cols] = te.fit_transform(target_df[cat_cols], target_df['Churn'])

