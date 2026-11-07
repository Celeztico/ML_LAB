'''

Investigate the bias-variance tradeoff using polynomial regression on the Boston Housing
dataset. Plot the training and validation errors for various polynomial degrees and discuss
the tradeoff between bias and variance.
Tasks:
● Load and preprocess the Boston Housing dataset.
● Implement polynomial regression with varying degrees.
● Plot training and validation errors for each degree.
● Discuss the bias-variance tradeoff and its impact on model performance.

'''


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("datasets/BostonHousing.csv")

X = df[["rm"]]
y = df["medv"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_errors = []
val_errors = []
degrees = range(1, 6)

# Test different polynomial degrees
for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)

    Xtr = poly.fit_transform(X_train)
    Xte = poly.transform(X_test)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    model = LinearRegression()
    model.fit(Xtr, y_train)

    train_pred = model.predict(Xtr)
    val_pred = model.predict(Xte)

    train_errors.append(mean_squared_error(y_train, train_pred))
    val_errors.append(mean_squared_error(y_test, val_pred))

    print("Degree:", d)
    print("Training Error:", train_errors[-1])
    print("Validation Error:", val_errors[-1])

# Plot errors
plt.plot(degrees, train_errors, marker="o", label="Training Error")
plt.plot(degrees, val_errors, marker="o", label="Validation Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.show()
