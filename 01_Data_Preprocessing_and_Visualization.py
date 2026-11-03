"""ML Assignment 1: Data Preprocessing and Visualization"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("tested.csv")

# 1. Statistical summary
df.describe()

# 2. Dataset information
df.info()

# 3. First 10 rows
df.head(10)

# 4. Last 10 rows
df.tail(10)

# Check missing values
df.isnull().sum()

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove remaining rows with missing values
df.dropna(inplace=True)

# 5. Verify missing values
df.isnull().sum()

# Encode categorical columns
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 6. View the last rows
df.tail()

# Scale the Fare column
scaler = StandardScaler()
numerical_cols = ["Fare"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 7. View the first 10 rows
df.head(10)

# 8. Plot Age by Sex
x = df["Sex"]
y = df["Age"]

plt.bar(x, y)
plt.xlabel("Sex")
plt.ylabel("Age")
plt.title("Age by Sex (Individual Data Points)")
plt.show()
