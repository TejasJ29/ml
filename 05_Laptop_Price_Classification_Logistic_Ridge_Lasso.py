"""ML Assignment 5: Laptop Price Classification and Regularization"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("laptop_price - dataset.csv")
df.head(5)

# 2. Statistical summary
df.describe()

# 3. Dataset information
df.info()

# 4. Check missing values
df.isnull().sum()

# 5. Create price classes using the average price
average_price = df["Price (Euro)"].mean()

df["Price_Class"] = 0
df.loc[df["Price (Euro)"] > average_price, "Price_Class"] = 1

x = df[["Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)"]]
y = df["Price_Class"]

print(df["Price_Class"].value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 6. Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

model_pred = model.predict(x_test)
logistic_accuracy = accuracy_score(y_test, model_pred)

print("Logistic Regression Accuracy:", logistic_accuracy)

# 7. Correlation matrix and heatmap
correlation = df[["Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)"]].corr()

print("Correlation Matrix:")
print(correlation)

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 8. Ridge Logistic Regression (L2 regularization)
ridge = LogisticRegression(penalty="l2", max_iter=1000)
ridge.fit(x_train, y_train)

ridge_pred = ridge.predict(x_test)
ridge_accuracy = accuracy_score(y_test, ridge_pred)

print("Ridge Accuracy:", ridge_accuracy)

# 9. Lasso Logistic Regression (L1 regularization)
lasso = LogisticRegression(
    penalty="l1", solver="liblinear", max_iter=1000
)
lasso.fit(x_train, y_train)

lasso_pred = lasso.predict(x_test)
lasso_accuracy = accuracy_score(y_test, lasso_pred)

print("Lasso Accuracy:", lasso_accuracy)

# 10. Compare model accuracy
plt.bar(
    ["Logistic", "Ridge", "Lasso"],
    [logistic_accuracy, ridge_accuracy, lasso_accuracy],
    color=["blue", "red", "green"]
)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.show()
