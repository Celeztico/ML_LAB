'''

9.Implement a Decision Tree classifier using the ID3 algorithm to segment customers based on their purchasing behavior using the Online Retail dataset. Analyze the tree structure and discuss the feature importance.
Tasks:
Load and preprocess the Online Retail dataset.
Implement Decision Tree using the ID3 algorithm.
Visualize the decision tree and analyze feature importance.
Discuss how the tree structure helps in understanding customer behavior.

'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_excel("datasets/Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Create customer features
data = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (df["InvoiceDate"].max() - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    Quantity=("Quantity", "sum"),
    Products=("StockCode", "nunique")
).reset_index()

# Create customer segments
data["Segment"] = pd.qcut(
    data["Monetary"], 3, labels=["Low", "Medium", "High"]
)

X = data[["Recency", "Frequency", "Quantity", "Products"]]
y = data["Segment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train ID3-style tree using entropy
model = DecisionTreeClassifier(
    criterion="entropy", max_depth=4, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Feature Importance:")
print(pd.Series(model.feature_importances_, index=X.columns))

# Plot decision tree
plt.figure(figsize=(14, 7))
plot_tree(model, feature_names=X.columns,
          class_names=model.classes_.astype(str),
          filled=True, rounded=True)
plt.show()

# Plot feature importance
pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(
    kind="barh", title="Feature Importance"
)
plt.show()