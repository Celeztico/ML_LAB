'''

14.Implement and compare the performance of a neural network using different activation
functions (Sigmoid, ReLU, Tanh) on the MNIST dataset. Analyze how each activation
function affects the training process and classification accuracy.
Tasks:
● Load and preprocess the MNIST dataset.
● Implement neural networks using Sigmoid, ReLU, and Tanh activation functions.
● Train and evaluate each network.
● Compare training times, convergence, and classification accuracy.

'''


import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load local dataset
train = pd.read_csv("datasets/mnist_train.csv").iloc[:10000]
test = pd.read_csv("datasets/mnist_test.csv").iloc[:2000]

X_train = train.drop("label", axis=1).values / 255.0
y_train = train["label"].values
X_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

# Compare activation functions
activations = ["logistic", "relu", "tanh"] # sigmoid = logistic in scikitlearn as compared to tensor flow
results = {}

for act in activations:
    model = MLPClassifier(hidden_layer_sizes=(128,),
                          activation=act,
                          max_iter=30,
                          random_state=42)

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[act] = acc

    print(act, "Accuracy:", acc,
          "Time:", round(elapsed, 2), "seconds")

# Plot comparison
plt.bar(results.keys(), results.values())
plt.xlabel("Activation Function")
plt.ylabel("Accuracy")
plt.title("MNIST Activation Comparison")
plt.show()
