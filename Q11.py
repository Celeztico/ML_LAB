'''

11.Implement a Linear Support Vector Machine (SVM) to classify the Iris dataset. Visualize the decision boundary and discuss how the margin is determined.
Tasks:
● Load and preprocess the Iris dataset.
● Implement a Linear SVM for binary classification (e.g., classify Setosa vs. Non-
Setosa).
● Visualize the decision boundary and margin.
● Discuss the concept of the margin and how it influences classification.

'''



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load local dataset
cols = ["sepal_length", "sepal_width", "petal_length",
        "petal_width", "species"]
df = pd.read_csv("datasets/iris.csv", names=cols).dropna()

# Binary classification: Setosa vs Non-Setosa
y = (df["species"] == "Iris-setosa").astype(int)
X = df[["petal_length", "petal_width"]]

# Scale features and train SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = SVC(kernel="linear", C=1)
model.fit(X, y)
pred = model.predict(X)

print("Accuracy:", accuracy_score(y, pred))

# Plot decision boundary and margins
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Automatically handles meshgrid, decision_function, and shading
DecisionBoundaryDisplay.from_estimator(
    model,
    X,
    plot_method="contour",
    levels=[-1, 0, 1],
    colors="k",
    linestyles=["--", "-", "--"],
)

# 2. Plot your data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")

# 3. Highlight support vectors (optional, if your lab asks for it)
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors="none",
    edgecolors="red",
)

plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Linear SVM: Iris Classification")
plt.show()
