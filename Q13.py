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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("datasets/winequality-red.csv", sep=";")
X = df.drop("quality", axis=1)
y = LabelEncoder().fit_transform(df["quality"])

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Try different MLP architectures
architectures = [[16], [32], [32, 16], [64, 32, 16]]
results = []

for layers in architectures:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))

    for neurons in layers:
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))

    model.add(tf.keras.layers.Dense(len(set(y)), activation="softmax"))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=30, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    results.append(acc)
    print("Layers:", layers, "Accuracy:", acc)

# Plot comparison
labels = [str(a) for a in architectures]
plt.bar(labels, results)
plt.xlabel("Hidden layers (neurons)")
plt.ylabel("Test accuracy")
plt.title("MLP Architecture Comparison")
plt.show()
