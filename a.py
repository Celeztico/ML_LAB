import pandas as pd
from sklearn.datasets import load_digits

d = load_digits()
df = pd.DataFrame(d.data)
df["target"] = d.target
df.to_csv("datasets/digits.csv", index=False)