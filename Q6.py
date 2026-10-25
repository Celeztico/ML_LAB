'''

6.Implement a logistic regression model to predict the likelihood of a disease using the Pima Indians Diabetes dataset. Compare the performance with and without feature scaling.
Tasks:
● Load and preprocess the Pima Indians Diabetes dataset.
● Implement logistic regression for binary classification.
● Evaluate model performance with and without feature scaling.
● Analyze metrics such as accuracy, precision, recall, and F1-score.

'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("datasets/pima_diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
for name, xtr, xte in [
    ("Without Scaling", X_train, X_test),
    ("With Scaling", X_train_scaled, X_test_scaled)
]:
    model = LogisticRegression(max_iter=5000)
    model.fit(xtr, y_train)
    pred = model.predict(xte)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))