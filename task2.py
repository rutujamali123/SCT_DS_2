# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Simulated dataset (customer purchase history)
# Columns: CustomerID, Annual Income (k$), Spending Score (1-100)
data = {
    'CustomerID': range(1, 11),
    'Annual_Income': [15, 16, 17, 18, 45, 46, 47, 85, 86, 87],
    'Spending_Score': [39, 81, 6, 77, 40, 42, 50, 76, 94, 50]
}
df = pd.DataFrame(data)

print("Customer Dataset:")
print(df)

# Step 3: Select features for clustering
X = df[['Annual_Income', 'Spending_Score']]

# Standardize data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # 3 clusters
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClustered Data:")
print(df)

# Step 5: Visualization
plt.figure(figsize=(8,6))
plt.scatter(df['Annual_Income'], df['Spending_Score'], 
            c=df['Cluster'], cmap='viridis', s=100)

# Plot cluster centers (reverse transform from scaled data)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X', label='Centroids')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()