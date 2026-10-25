'''

10.Implement and compare Logistic Regression and Decision Trees on the Adult Income dataset for predicting income levels. Evaluate both models based on performance metrics and interpretability.
Tasks:
Load and preprocess the Adult Income dataset.
Implement both Logistic Regression and Decision Trees.
Compare the models based on metrics such as accuracy, precision, recall, and F1-score.
Discuss the interpretability of both models and their suitability for the dataset.

'''



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load local dataset
cols = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"]

df = pd.read_csv("datasets/adult.csv", names=cols, skipinitialspace=True, na_values="?")
df = df.dropna()
df["income"] = df["income"].str.replace(".", "", regex=False).map({"<=50K": 0, ">50K": 1})

X = df.drop("income", axis=1)
y = df["income"]

num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and compare models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
}

results = []

for name, model in models.items():
    pipe = make_pipeline(pre, model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    scores = [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0)
    ]
    results.append(scores)
    print(name, "Accuracy, Precision, Recall, F1:", scores)

# Plot comparison
result_df = pd.DataFrame(
    results, index=models.keys(),
    columns=["Accuracy", "Precision", "Recall", "F1-score"]
)
result_df.plot(kind="bar", ylim=(0, 1), figsize=(8, 5))
plt.title("Logistic Regression vs Decision Tree")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Visualize decision tree
tree_pipe = make_pipeline(
    pre, DecisionTreeClassifier(max_depth=3, random_state=42)
)
tree_pipe.fit(X_train, y_train)

plt.figure(figsize=(16, 7))
plot_tree(
    tree_pipe.named_steps["decisiontreeclassifier"],
    max_depth=3, filled=True
)
plt.title("Decision Tree for Adult Income")
plt.show()
