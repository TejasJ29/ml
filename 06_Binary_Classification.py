"""ML Assignment 6: Heart Disease Classification using Logistic Regression"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("heart_disease.csv")

df.head()
df.info()
df.describe()

# Handle missing numerical values
df = df.fillna(df.median(numeric_only=True))
print(df.isnull().sum())

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target
x = df.drop(["Heart Disease Status"], axis=1)
y = df["Heart Disease Status"]

# Split dataset while preserving class proportions
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(x_train, y_train)

model_predict = model.predict(x_test)

# Confusion matrix
cm = confusion_matrix(y_test, model_predict)

print("Confusion Matrix:")
print(cm)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.xticks([0, 1], ["No", "Yes"])
plt.yticks([0, 1], ["No", "Yes"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# Evaluation metrics
precision = precision_score(y_test, model_predict)
print("Precision:", precision)

recall = recall_score(y_test, model_predict)
print("Recall:", recall)

f1 = f1_score(y_test, model_predict)
print("F1 Score:", f1)
