'''

Implement bootstrapping and cross-validation on the Iris dataset. Compare the model
performance metrics (e.g., accuracy, F1-score) obtained using these resampling methods.
Discuss the advantages and disadvantages of each method.
Tasks:
● Load and preprocess the Iris dataset.
● Implement bootstrapping to generate multiple samples and evaluate the model.
● Implement k-fold cross-validation and evaluate the model.
● Compare the performance metrics and discuss the pros and cons of each resampling method.

'''

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# Load local Iris dataset
cols = ["sepal_length", "sepal_width", "petal_length",
        "petal_width", "species"]
df = pd.read_csv("datasets/iris.csv", names=cols).dropna()

X = df.drop("species", axis=1)
y = df["species"]

# 1. Bootstrapping
acc, f1 = [], []

for i in range(10):
    sample = np.random.choice(len(X), len(X), replace=True)

    X_train = X.iloc[sample]
    y_train = y.iloc[sample]

    X_test = X.drop(X.index[sample])
    y_test = y.drop(y.index[sample])

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc.append(accuracy_score(y_test, pred))
    f1.append(f1_score(y_test, pred, average="weighted"))

print("Bootstrap Accuracy:", np.mean(acc))
print("Bootstrap F1 Score:", np.mean(f1))

# 2. K-Fold Cross Validation
model = DecisionTreeClassifier()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

print("K-Fold Accuracy:", scores.mean())
print("K-Fold F1 Score:",
      cross_val_score(model, X, y, cv=kf, scoring="f1_weighted").mean())