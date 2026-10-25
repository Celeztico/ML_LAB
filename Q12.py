'''

12.Implement and compare the performance of SVM classifiers with linear, polynomial, and RBF kernels on the Fashion MNIST dataset. Analyze the advantages and disadvantages of each kernel type.
Tasks:
● Load and preprocess the Fashion MNIST dataset.
● Implement SVM with linear, polynomial, and RBF kernels.
● Compare the classification performance for each kernel.
● Discuss the strengths and weaknesses of each kernel type.

'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Load local CSV files
train = pd.read_csv("datasets/fashion_train.csv").sample(
    5000, random_state=42
)
test = pd.read_csv("datasets/fashion_test.csv").sample(
    1000, random_state=42
)

X_train = train.drop("label", axis=1) / 255.0
y_train = train["label"]
X_test = test.drop("label", axis=1) / 255.0
y_test = test["label"]

# SVM kernels
models = {
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=3),
    "RBF": SVC(kernel="rbf")
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results[name] = accuracy_score(y_test, pred)
    print(name)
    print("Accuracy:", results[name])
    print("F1-score:", f1_score(y_test, pred, average="weighted"))

# Plot accuracy comparison
plt.bar(results.keys(), results.values())
plt.title("SVM Kernel Comparison")
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
