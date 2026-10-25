'''

8.Implement the K-Nearest Neighbors (KNN) algorithm for image classification using the Fashion MNIST dataset. Experiment with different values of K and analyze their impact on model performance.
Tasks:
● Load and preprocess the Fashion MNIST dataset.
● Implement KNN for multi-class classification.
● Experiment with different values of K and evaluate performance.
● Discuss the impact of different K values on model accuracy and computational efficiency.

'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load local CSV files
train = pd.read_csv("datasets/fashion_train.csv")
test = pd.read_csv("datasets/fashion_test.csv")

X_train = train.drop("label", axis=1).iloc[:10000] / 255.0
y_train = train["label"].iloc[:10000]
X_test = test.drop("label", axis=1).iloc[:2000] / 255.0
y_test = test["label"].iloc[:2000]

# Try different K values
k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print("K =", k, "Accuracy =", acc)

# Plot results
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracy vs K")
plt.xticks(k_values)
plt.grid(True)
plt.show()
