'''

4.Estimate the parameters of a logistic regression model using MLE and MAP on the Breast Cancer Wisconsin dataset. Compare the results and discuss the effects of regularization.
Tasks:
● Load and preprocess the dataset.
● Implement logistic regression with MLE.
● Apply MAP estimation with different regularization priors (L1 and L2 regularization).
● Compare the performance and parameter estimates with MLE and MAP.

'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("datasets/breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "MLE": LogisticRegression(penalty=None, max_iter=5000),
    "MAP-L1": LogisticRegression(penalty="l1", solver="liblinear", C=1),
    "MAP-L2": LogisticRegression(penalty="l2", solver="lbfgs", C=1)
}

# Train and compare
for name, model in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))
    print("Coefficients:", pipe.named_steps["model"].coef_[0][:5])