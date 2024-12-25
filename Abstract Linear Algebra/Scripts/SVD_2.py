import numpy as np

# Create a matrix
matrix = np.random.rand(1000, 500)

# Calculate SVD
u, s, vh = np.linalg.svd(matrix)

# Reconstruct the matrix
s_matrix = np.zeros((u.shape[1], vh.shape[0]))
np.fill_diagonal(s_matrix, s)
reconstructed_matrix = u @ s_matrix @ vh

# Print the original and reconstructed matrix shapes
print(f"Original matrix shape: {matrix.shape}")
print(f"Reconstructed matrix shape: {reconstructed_matrix.shape}")
