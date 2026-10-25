'''

5.Use MLE and MAP to estimate the parameters of a multinomial distribution on the 20 Newsgroups dataset. Explore the impact of different priors on the estimation.
Tasks:
● Load and preprocess the dataset.
● Implement MLE for multinomial distribution parameter estimation.
● Apply MAP estimation with various priors (e.g., Dirichlet priors).
● Compare results and evaluate the effect of different priors

'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
df = pd.read_csv("datasets/20newsgroups.csv")
texts = df["text"].dropna()

# Convert text to word counts
vectorizer = CountVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(texts)

# MLE
counts = np.asarray(X.sum(axis=0)).ravel()
mle = counts / counts.sum()


# MAP with different Dirichlet priors
def map_estimate(alpha):
    return (counts + alpha - 1) / (counts.sum() + len(counts) * (alpha - 1))


map_01 = map_estimate(1.1)
map_1 = map_estimate(2)
map_10 = map_estimate(11)

# Compare first 20 word probabilities
results = pd.DataFrame({
    "Word": vectorizer.get_feature_names_out()[:20],
    "MLE": mle[:20],
    "MAP alpha=1.1": map_01[:20],
    "MAP alpha=2": map_1[:20],
    "MAP alpha=11": map_10[:20]
})

print(results)

# Compare difference from MLE
print("\nDifference from MLE:")
print("Alpha=1.1:", np.sum(np.abs(mle - map_01)))
print("Alpha=2:", np.sum(np.abs(mle - map_1)))
print("Alpha=11:", np.sum(np.abs(mle - map_10)))