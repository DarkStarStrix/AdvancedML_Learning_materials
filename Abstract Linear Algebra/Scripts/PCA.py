import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# visualize the reduced data
def plot_reduced_data(X_reduced, y):
    plt.figure (figsize=(8, 6))
    for i in range (3):
        plt.scatter (X_reduced [y == i, 0], X_reduced [y == i, 1], label=f"Class {i}", alpha=0.7)
    plt.xlabel ("Component 1")
    plt.ylabel ("Component 2")
    plt.title ("Iris Dataset - Reduced Data")
    plt.legend ()
    plt.show ()


def pca(X, n_components):
    """
  Performs PCA on the given data matrix.

  Args:
    X: Data matrix with samples as rows and features as columns.
    n_components: The number of principal components to keep.

  Returns:
    X_transformed: The transformed data in the reduced dimensionality space.
    explained_variance_ratio: The percentage of variance explained by each principal component.
  """

    # Standardize the data
    X_standardized = (X - np.mean (X, axis=0)) / np.std (X, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov (X_standardized, rowvar=False)

    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig (covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort (eigenvalues) [::-1]
    sorted_eigenvectors = eigenvectors [:, sorted_indices]

    # Select the top n_components eigenvectors
    projection_matrix = sorted_eigenvectors [:, :n_components]

    # Project the data onto the principal components
    X_transformed = X_standardized.dot (projection_matrix)

    explained_variance = eigenvalues [sorted_indices]
    explained_variance_ratio = explained_variance / np.sum (eigenvalues)
    return X_transformed, explained_variance_ratio [:n_components]


iris = datasets.load_iris ()
X = iris.data

# Apply PCA to reduce to 2 dimensions
X_transformed, explained_variance_ratio = pca (X, 2)

print ("Transformed Data:\n", X_transformed [:5, :])  # Print the first 5 rows of transformed data
print ("\nExplained Variance Ratio:\n", explained_variance_ratio)
plot_reduced_data (X_transformed, iris.target)
