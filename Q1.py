'''
Q1 Implement linear regression with one variable on the california housing dataset to predict housing prices based on a single feature (eg avrooms per dwelling)
Tasks:
load and preprocess the datas.
implement linear regression using both gradient descent and the normal equation
 evaluate the model performance using metrics such as mean squared error and R squared.
Visualize the fitted line along with the data points.
'''


import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("datasets/california_housing.csv")

df["AveRooms"] = df["total_rooms"] / df["households"]


# feature and target
X = df[['AveRooms']]
y = df['median_house_value'] / 100000

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# gradient descent

gd_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
])

gd_model.fit(X_train, y_train)
y_pred_gd = gd_model.predict(X_test)

print("Gradient Descent")
print("MSE: ", mean_squared_error(y_test, y_pred_gd))
print("R2 score: ", r2_score(y_test, y_pred_gd))

# Nomral equation

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ("model", LinearRegression())
])

lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nNormal equation")
print("MSE: ", mean_squared_error(y_test, y_pred_lr))
print("R2 score: ", r2_score(y_test, y_pred_lr))

# visualisation
idx = X_test["AveRooms"].argsort()

plt.scatter(X_test["AveRooms"], y_test, s=10)
plt.plot(X_test["AveRooms"].iloc[idx],
         y_pred_lr[idx], color="red")
plt.xlabel("Average Rooms per Dwelling")
plt.ylabel("Median House Value")
plt.title("Linear Regression")
plt.show()
