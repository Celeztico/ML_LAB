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
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2)
plt.contour(xx, yy, Z, levels=[-1, 0, 1],
            colors="black", linestyles=["--", "-", "--"])

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black")
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=150, facecolors="none", edgecolors="red")

plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Linear SVM: Iris Classification")
plt.show()
