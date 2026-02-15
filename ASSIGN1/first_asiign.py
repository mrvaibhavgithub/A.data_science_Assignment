# ---------------------------------------------------------
# 1. Import libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 2. Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("employee_records.csv")
print("Dataset loaded successfully")
print("=" * 100)

# ---------------------------------------------------------
# 3. Clean column names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()
print("Column names after cleaning:")
print(df.columns.tolist())
print("=" * 100)

# ---------------------------------------------------------
# 4. Dataset exploration
# ---------------------------------------------------------
print("First 5 records:")
print(df.head())
print("=" * 100)

print("Dataset shape (rows, columns):")
print(df.shape)
print("=" * 100)

print("Dataset information:")
df.info()
print("=" * 100)

# ---------------------------------------------------------
# 5. Check missing values and zero values
# ---------------------------------------------------------
print("Missing values in each column:")
print(df.isnull().sum())
print("=" * 100)

numerical_cols = df.select_dtypes(include=np.number).columns
print("Zero values in numerical columns:")
print((df[numerical_cols] == 0).sum())
print("=" * 100)

# ---------------------------------------------------------
# 6. Remove duplicate records
# ---------------------------------------------------------
print("Duplicate records:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates removed")
print("=" * 100)

# ---------------------------------------------------------
# 7. Handle missing values
# ---------------------------------------------------------
# Numerical → Mean
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Categorical → Mode
cat_cols = df.select_dtypes(include=['object', 'string', 'bool']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled")
print("=" * 100)

# ---------------------------------------------------------
# 8. Salary analysis
# ---------------------------------------------------------
print("Salary Statistics")
print("Mean Salary:", df['salary'].mean())
print("Median Salary:", df['salary'].median())
print("Mode Salary:", df['salary'].mode()[0])
print("Salary Skewness:", df['salary'].skew())
print("=" * 100)

# ---------------------------------------------------------
# 9. Salary Distribution Plot
# ---------------------------------------------------------
plt.figure()
sns.histplot(df['salary'], kde=True)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()
