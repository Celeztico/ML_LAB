'''

Implement and compare hierarchical (agglomerative) and partitional (K-means) clustering
algorithms on the Mall Customers dataset. Discuss the strengths and weaknesses of each
method based on clustering results and evaluation metrics.
Tasks:
● Load and preprocess the Mall Customers dataset.
● Apply both hierarchical (agglomerative) and K-means clustering.
● Compare results using metrics such as inertia, silhouette score, and clustering visualization.
● Discuss the advantages and disadvantages of each clustering method.

'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load local dataset
df = pd.read_csv("datasets/Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize
X = StandardScaler().fit_transform(X)

# K-Means clustering
km = KMeans(n_clusters=5, random_state=42, n_init=10)
k_labels = km.fit_predict(X)

# Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5)
h_labels = hc.fit_predict(X)

# Evaluation
print("K-Means Inertia:", km.inertia_)
print("K-Means Silhouette:", silhouette_score(X, k_labels))

print("Hierarchical Silhouette:", silhouette_score(X, h_labels))

# Visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=k_labels, cmap="rainbow")
plt.title("K-Means Clustering")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=h_labels, cmap="rainbow")
plt.title("Hierarchical Clustering")

plt.show()