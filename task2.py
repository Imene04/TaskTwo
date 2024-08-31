import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sample synthetic dataset
data = {
    'customer_id': [1, 2, 3, 4, 5, 6],
    'total_purchase_amount': [500, 700, 1500, 800, 2000, 300],
    'number_of_purchases': [5, 7, 15, 8, 20, 3],
    'days_since_last_purchase': [10, 20, 5, 30, 2, 50]
}

df = pd.DataFrame(data)

# Selecting the features for clustering
X = df[['total_purchase_amount', 'number_of_purchases', 'days_since_last_purchase']]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 7):  # Adjusted to not exceed the number of samples
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 7), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-means with an appropriate number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adding cluster labels to the original dataset
df['Cluster'] = clusters

# Visualizing the clusters
plt.scatter(X_scaled[clusters == 0, 0], X_scaled[clusters == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[clusters == 1, 0], X_scaled[clusters == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[clusters == 2, 0], X_scaled[clusters == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Total Purchase Amount (scaled)')
plt.ylabel('Number of Purchases (scaled)')
plt.legend()
plt.show()
