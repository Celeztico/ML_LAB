import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("datasets/california_housing.csv")
df["AveRooms"] = df["total_rooms"] / df["households"]

X = df[["AveRooms"]].values
y = df["median_house_value"].values / 100000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization using function
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add intercept column
Xtr = np.c_[np.ones(len(X_train)), X_train]
Xte = np.c_[np.ones(len(X_test)), X_test]

# 1. Gradient Descent
theta = np.zeros(2)
etha = 0.01

for i in range(1000):
    error = Xtr @ theta - y_train
    theta -= etha * (Xtr.T @ error) / len(y_train)

y_pred_gd = Xte @ theta
print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2:", r2_score(y_test, y_pred_gd))

# 2. Normal Equation
theta = np.linalg.pinv(Xtr.T @ Xtr) @ Xtr.T @ y_train
y_pred_ne = Xte @ theta

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R2:", r2_score(y_test, y_pred_ne))

# Plot
idx = np.argsort(X_test[:, 0])
plt.scatter(X_test[:, 0], y_test, s=10)
plt.plot(X_test[idx, 0], y_pred_ne[idx], "r")
plt.xlabel("Standardized Average Rooms")
plt.ylabel("Median House Value (100,000s)")
plt.title("Linear Regression")
plt.show()