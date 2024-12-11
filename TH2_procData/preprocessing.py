import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv("bank-data.csv")

print("Du lieu ban dau")
print(df.head())

df.replace('?', np.nan, inplace=True)

# Delete att
if 'id' in df.columns: df = df.drop(columns=["id"])
print("Sau khi xoa cot ID")
 
# Discretization
if 'age' in df.columns:
    df['age_bins'] = pd.cut(df['age'], bins=3, labels=["young", "mid", "old"])

if 'income' in df.columns:
    df['income_bins'] = pd.cut(df['income'], bins=3, labels=["low", "medium", "high"])
print("Sau khi roi rac hoa")

# Missing values
missing_cols = df.columns[df.isnull().any()]
print(f"Cac cot du lieu bi thieu: {missing_cols}")

missing_counts = df[missing_cols].isnull().sum()
print("So luong gia tri thieu trong tung cot:")
print(missing_counts)

# cate att
cate_cols = df.select_dtypes(include=["object", "category"]).columns
imputer_cat = SimpleImputer(strategy="most_frequent")
df[cate_cols] = imputer_cat.fit_transform(df[cate_cols])

# digit att
num_cols = df.select_dtypes(include=["number"]).columns
imputer_num = SimpleImputer(strategy="mean")
df[num_cols] = imputer_num.fit_transform(df[num_cols])

df.to_csv("processed_data.csv", index=False)