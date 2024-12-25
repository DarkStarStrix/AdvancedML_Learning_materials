import jax.numpy as jnp
import jax

# Create two matrices
matrix_a = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
matrix_b = jax.random.normal(jax.random.PRNGKey(1), (1000, 1000))

# Perform matrix multiplication
result = jnp.matmul(matrix_a, matrix_b)

print(result)
