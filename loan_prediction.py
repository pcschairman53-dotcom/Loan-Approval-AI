# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load Dataset
# Google Colab-এ loan_data.csv আপলোড করুন
df = pd.read_csv("Loan_data - Sheet1.csv")
print(df.columns.tolist())
# Step 3: Encode Categorical Columns
encoder = LabelEncoder()

categorical_cols = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Loan _Status"  # Corrected column name: 'Loan _Status' instead of 'Loan_Status' and removed 'Property_Area'
]

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Missing Values Fill
df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 4: Features & Target
X = df.drop(["Loan _Status", "Loan_ID"], axis=1) # Corrected column name
y = df["Loan _Status"] # Corrected column name

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")
import matplotlib.pyplot as plt

df["Loan_Status"].value_counts().plot(kind="bar")
plt.title("Loan Approval Count")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df["ApplicantIncome"], bins=10)
plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df["LoanAmount"], bins=10)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()
df["Property_Area"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Property Area Distribution")
plt.ylabel("")
plt.show()
import numpy as np

corr = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()
plt.figure(figsize=(8,5))
plt.boxplot(df["ApplicantIncome"])
plt.title("Applicant Income Box Plot")
plt.show()
plt.figure(figsize=(8,5))
plt.scatter(df["ApplicantIncome"], df["LoanAmount"])
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.title("Income vs Loan Amount")
plt.show()