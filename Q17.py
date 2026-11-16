'''

17.Implement and apply K-means clustering to the Digits dataset. Experiment with different numbers of clusters and evaluate the clustering results using metrics such as inertia and silhouette score. Analyze how the choice of K affects clustering performance.
Tasks:
● Load and preprocess the Digits dataset.
● Implement K-means clustering with various numbers of clusters.
● Evaluate clustering performance using inertia and silhouette score.
● Analyze the impact of the number of clusters on clustering quality.

'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load local dataset
df = pd.read_csv("datasets/digits.csv")
X = df.drop("target", axis=1)

# Preprocess
X = StandardScaler().fit_transform(X)

inertia = []
silhouette = []
K = range(2, 16)

# K-Means for different K values
for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X, labels))

# Plot results
plt.subplot(1, 2, 1)
plt.plot(K, inertia, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.subplot(1, 2, 2)
plt.plot(K, silhouette, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")

plt.tight_layout()
plt.show()

# doesnt work for digits only for 2 feature models
# best_k = 10 # Replace with whatever the elbow/silhouette shows is best
# best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
# best_labels = best_model.fit_predict(X_s)
#
# plt.figure()
# # Plots the data points colored by cluster
# plt.scatter(X_s[:, 0], X_s[:, 1], c=best_labels, cmap='viridis', edgecolors='black')
# # Plots the massive red 'X' for the center of each cluster
# plt.scatter(best_model.cluster_centers_[:, 0], best_model.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
# plt.title('Cluster Scatter Plot')
# plt.legend()