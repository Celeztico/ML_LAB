'''

Implement bagging and boosting ensemble methods on the Titanic dataset. Compare the
performance of both methods in terms of accuracy, precision, recall, and F1-score.
Discuss how each method improves model performance and their respective strengths and
weaknesses.
Tasks:
● Load and preprocess the Titanic dataset.
● Implement bagging using a base classifier (e.g., decision tree) and evaluate performance.
● Implement boosting using a boosting algorithm (e.g., AdaBoost) and evaluate performance.
● Compare performance metrics and discuss the strengths and weaknesses of each method.

'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("datasets/titanic.csv")

# Select features
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
df = df.dropna()

# Convert categorical data
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models
models = {
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42
    ),
    "Boosting": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50, random_state=42
    )
}

results = []

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    results.append([acc, pre, rec, f1])

    print("\n", name)
    print("Accuracy:", acc)
    print("Precision:", pre)
    print("Recall:", rec)
    print("F1 Score:", f1)

# Compare results
result_df = pd.DataFrame(
    results, index=["Bagging", "Boosting"],
    columns=["Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nComparison:\n", result_df)

result_df.plot(kind="bar", ylim=(0, 1), title="Bagging vs Boosting")
plt.ylabel("Score")
plt.show()