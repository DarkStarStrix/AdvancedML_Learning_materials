import jax.numpy as jnp
import jax

# Create a matrix
matrix = jax.random.normal(jax.random.PRNGKey(0), (1000, 500))

# Calculate SVD
u, s, vh = jnp.linalg.svd(matrix)

# Reconstruct the matrix
s_matrix = jnp.zeros((u.shape[1], vh.shape[0]))
s_matrix = s_matrix.at[:s.shape[0], :s.shape[0]].set(jnp.diag(s))

reconstructed_matrix = jnp.dot(u, jnp.dot(s_matrix, vh))

# Print the original and reconstructed matrix shapes
print(f"Original matrix shape: {matrix.shape}")
print(f"Reconstructed matrix shape: {reconstructed_matrix.shape}")
