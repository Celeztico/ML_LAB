'''

13.Implement and train a Multilayer Feed-Forward Network (MLP) on the Wine Quality
dataset. Experiment with different numbers of hidden layers and neurons, and discuss how
these choices affect the network’s performance.
Tasks:
● Load and preprocess the Wine Quality dataset.
● Design and implement an MLP with varying architectures (different hidden layers
and neurons).
● Train and evaluate the network.
● Discuss the impact of architecture choices on performance.

'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("datasets/winequality-red.csv", sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Different MLP architectures
architectures = [(16,), (32,), (32, 16), (64, 32, 16)]
results = []

for layers in architectures:
    model = MLPClassifier(hidden_layer_sizes=layers,
                          max_iter=500, random_state=42)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    results.append(acc)

    print("Layers:", layers, "Accuracy:", acc)

# Plot comparison
plt.bar([str(a) for a in architectures], results)
plt.xlabel("Hidden Layers")
plt.ylabel("Accuracy")
plt.title("MLP Architecture Comparison")
plt.show()