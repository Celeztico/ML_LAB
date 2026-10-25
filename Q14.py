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
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score

# Load local dataset
train = pd.read_csv("datasets/mnist_train.csv").iloc[:10000]
test = pd.read_csv("datasets/mnist_test.csv").iloc[:2000]

X_train = train.drop("label", axis=1).values / 255.0
y_train = train["label"].values
X_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

results = {}

# Compare activation functions
for act in ["sigmoid", "relu", "tanh"]:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation=act),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    start = time.time()
    model.fit(X_train, y_train, epochs=5, verbose=0)
    elapsed = time.time() - start

    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, pred)
    results[act] = acc

    print(act, "Accuracy:", acc, "Time:", round(elapsed, 2), "seconds")

# Plot comparison
plt.bar(results.keys(), results.values())
plt.xlabel("Activation Function")
plt.ylabel("Test Accuracy")
plt.title("MNIST Activation Function Comparison")
plt.show()
