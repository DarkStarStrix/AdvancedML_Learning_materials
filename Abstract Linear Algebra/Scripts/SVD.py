import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Write a visualization function to plot the reduced data
def plot_reduced_data(X_reduced, y):
    plt.figure (figsize=(8, 6))
    for i in range (3):
        plt.scatter (X_reduced [y == i, 0], X_reduced [y == i, 1], label=f"Class {i}", alpha=0.7)
    plt.xlabel ("Component 1")
    plt.ylabel ("Component 2")
    plt.title ("Iris Dataset - Reduced Data")
    plt.legend ()
    plt.show ()


# Load the Iris dataset
iris = load_iris ()
X = iris.data

# Standardize the data (important for SVD)
scaler = StandardScaler ()
X_scaled = scaler.fit_transform (X)

# Compute the SVD
U, S, VT = np.linalg.svd (X_scaled)

# Choose the number of components to keep (e.g., 2 for visualization)
k = 2

# Reconstruct the data with reduced dimensions
X_reduced = U [:, :k] @ np.diag (S [:k]) @ VT [:k, :]

# Print the shape of the reduced data
print (f"Original data shape: {X_scaled.shape}")
print (f"Reduced data shape: {X_reduced.shape}"
         f"\n\nReduced data:\n{X_reduced}")
plot_reduced_data (X_reduced, iris.target)

# The X_reduced matrix now represents the Iris dataset in a lower-dimensional space.
# Visualize the reduced data using a scatter plot
