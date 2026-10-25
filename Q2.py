'''
2. Implement polynomial regression on the Auto MPG dataset to predict miles per gallon MPG based on engine displacement. Compare polynomial regression results with linear regression.
Tasks:
Load and preprocess the dataset,
implement polynomial regression of varying degrees,
compare the polynomial regression models with linear regression using metrics such as MSE and R-squared,
Visualise the polynomial fit
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('datasets/auto-mpg.csv')
df = df[["displacement", "mpg"]].replace("?", np.nan)
df = df.dropna()
df = df.astype(float)

X = df[["displacement"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLinear Regression")
print("-------------------------")
print("MSE :", mean_squared_error(y_test, y_pred_lr))
print("R2 Score :", r2_score(y_test, y_pred_lr))

degrees = [2, 3, 4, 5]

poly_models = {}

for degree in degrees:

    poly_model = Pipeline([
        ("polynomial", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])

    poly_model.fit(X_train, y_train)

    prediction = poly_model.predict(X_test)

    mse = mean_squared_error(y_test, prediction)

    r2 = r2_score(y_test, prediction)

    poly_models[degree] = poly_model

    print("\nDegree:", degree)
    print("MSE :", mse)
    print("R2 Score :", r2)

plt.figure(figsize=(10,6))
plt.scatter(X,y,color="blue", alpha=0.5, label="actual data")

X_curve = pd.DataFrame(
    {
        "displacement": np.linspace(
            X["displacement"].min(),
            X["displacement"].max(),
            300
        )
    }
)

# Linear regression curve

plt.plot(
    X_curve["displacement"],
    lr_model.predict(X_curve),
    color="red",
    linewidth=2,
    label="Linear Regression"
)



# Polynomial curves

colors = [
    "green",
    "orange",
    "purple",
    "black"
]


for degree, color in zip(degrees, colors):

    plt.plot(
        X_curve["displacement"],
        poly_models[degree].predict(X_curve),
        color=color,
        linewidth=2,
        label=f"Polynomial Degree {degree}"
    )


plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear vs Polynomial Regression on Auto MPG Dataset")

plt.legend()
plt.grid(True)

plt.show()
