'''

Implement and perform hyperparameter tuning for a neural network on the Fashion
MNIST dataset. Experiment with different learning rates, batch sizes, and epochs, and
discuss the impact on model performance.
Tasks:
● Load and preprocess the Fashion MNIST dataset.
● Experiment with different hyperparameters (learning rate, batch size, epochs).
● Train and evaluate the network.
● Discuss how hyperparameter choices affect model performance.

'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

# Load local dataset
train = pd.read_csv("datasets/fashion_train.csv").iloc[:10000]
test = pd.read_csv("datasets/fashion_test.csv").iloc[:2000]

X_train = train.drop("label", axis=1).values / 255.0
y_train = train["label"].values
X_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

# Hyperparameters
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [20, 50]

results = []

# Experiment with different values
for lr in learning_rates:
    for batch in batch_sizes:
        for ep in epochs:

            model = MLPClassifier(
                hidden_layer_sizes=(128,),
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=ep,
                random_state=42
            )

            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            results.append(acc)

            print("LR:", lr, "Batch:", batch,
                  "Epochs:", ep, "Accuracy:", acc,
                  "Time:", round(elapsed, 2))

# Plot results
plt.plot(results, marker="o")
plt.xlabel("Experiment")
plt.ylabel("Accuracy")
plt.title("Fashion MNIST Hyperparameter Tuning")
plt.show()