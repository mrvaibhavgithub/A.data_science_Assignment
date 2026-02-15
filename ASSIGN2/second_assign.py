# ---------------------------------------------------------
# 1. Import Libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("employee_records.csv")
print("Dataset Loaded Successfully")
print("=" * 80)

# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()
print("Column Names:")
print(df.columns.tolist())
print("=" * 80)

# ---------------------------------------------------------
# 4. Basic Dataset Information
# ---------------------------------------------------------
print("First 5 Records:")
print(df.head())
print("=" * 80)

print("Dataset Shape:")
print(df.shape)
print("=" * 80)

print("Missing Values in Each Column:")
print(df.isnull().sum())
print("=" * 80)

# ---------------------------------------------------------
# 5. Remove Duplicate Records
# ---------------------------------------------------------
df.drop_duplicates(inplace=True)
print("Duplicate Records Removed")
print("=" * 80)

# ---------------------------------------------------------
# 6. Handle Missing Values
# ---------------------------------------------------------
# Numerical columns → Mean
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Categorical columns → Mode
cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values Handled Successfully")
print("=" * 80)

# ---------------------------------------------------------
# 7. Select Numerical Columns for Scaling
# ---------------------------------------------------------
numeric_data = df[numerical_cols]

# ---------------------------------------------------------
# 8. Normalization (Min-Max Scaling)
# ---------------------------------------------------------
minmax_scaler = MinMaxScaler()
normalized_data = minmax_scaler.fit_transform(numeric_data)

normalized_df = pd.DataFrame(normalized_data, columns=numeric_data.columns)

print("Normalized Data (First 5 Rows):")
print(normalized_df.head())
print("=" * 80)

# ---------------------------------------------------------
# 9. Standardization (Z-Score Scaling)
# ---------------------------------------------------------
standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(numeric_data)

standardized_df = pd.DataFrame(standardized_data, columns=numeric_data.columns)

print("Standardized Data (First 5 Rows):")
print(standardized_df.head())
print("=" * 80)

print("Data Handling Completed Successfully")
