import jax.numpy as jnp
from jax import jit, grad
from jax.random import normal, PRNGKey
import numpy as np
import matplotlib.pyplot as plt

# Generate a random matrix
key = PRNGKey (0)
A = normal (key, (1000, 500))

# Initialize factor matrices
k = 100
U = normal (key, (1000, k))
V = normal (key, (500, k))


# Define the loss function (e.g., mean squared error)
@jit
def loss_fn(U, V, A):
    return jnp.mean ((A - jnp.dot (U, V.T)) ** 2)


# Calculate gradients
grad_U = grad (loss_fn, argnums=0)
grad_V = grad (loss_fn, argnums=1)


# Update rule (e.g., gradient descent)
@jit
def update(U, V, A, learning_rate):
    U = U - learning_rate * grad_U (U, V, A)
    V = V - learning_rate * grad_V (U, V, A)
    return U, V


# Optimization loop
learning_rate = 0.001
for _ in range (1000):
    U, V = update (U, V, A, learning_rate)

# Compute the final loss
final_loss = loss_fn (U, V, A)
print (final_loss)


# Visualize the factor matrices
def plot_factor_matrices(U, V):
    plt.figure (figsize=(8, 6))
    plt.subplot (1, 2, 1)
    plt.imshow (U, aspect='auto')
    plt.title ('Factor Matrix U')
    plt.subplot (1, 2, 2)
    plt.imshow (V.T, aspect='auto')
    plt.title ('Factor Matrix V')
    plt.show ()


plot_factor_matrices (U, V)
# The factor matrices U and V represent the low-rank approximation of the original matrix A.
