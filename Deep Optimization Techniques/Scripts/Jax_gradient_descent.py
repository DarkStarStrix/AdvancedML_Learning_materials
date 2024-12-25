import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit


def gradient_descent(loss_fn, params, learning_rate, num_iterations, data):
    """
    Implements gradient descent using JAX.

    Args:
        loss_fn: The loss function to minimize.
        params: Initial model parameters.
        learning_rate: The step size for each iteration.
        num_iterations: The number of iterations to perform.
        data: The training data.

    Returns:
        The optimized model parameters.
    """
    grad_fn = jax.grad (loss_fn)  # Automatic differentiation

    @jit
    def update(params, data):
        gradient = grad_fn (params, data)
        return params - learning_rate * gradient  # Update parameters

    for _ in range (num_iterations):
        params = update (params, data)
    return params


def train_model(loss_fn, init_params, learning_rate, num_iterations, data):
    """
    Train the model using gradient descent.

    Args:
        loss_fn: The loss function to minimize.
        init_params: Initial model parameters.
        learning_rate: The step size for each iteration.
        num_iterations: The number of iterations to perform.
        data: The training data.

    Returns:
        The optimized model parameters and a list of loss values.
    """
    params = init_params
    loss_values = []

    grad_fn = jax.grad (loss_fn)

    @jit
    def update(params, data):
        gradient = grad_fn (params, data)
        return params - learning_rate * gradient

    for i in range (num_iterations):
        params = update (params, data)
        loss = loss_fn (params, data)
        loss_values.append (loss)
        if i % 100 == 0:
            print (f"Iteration {i}, Loss: {loss}")

    return params, loss_values


def plot_loss(loss_values):
    """
    Plot the loss values over iterations.

    Args:
        loss_values: List of loss values.
    """
    plt.figure (figsize=(10, 5))
    plt.plot (loss_values)
    plt.title ('Loss over Iterations')
    plt.xlabel ('Iteration')
    plt.ylabel ('Loss')
    plt.show ()


# Define the loss function
def loss_fn(params, data):
    X, y = data
    predictions = jnp.dot (X, params)
    return jnp.mean ((predictions - y) ** 2)


# Initialize parameters, learning rate, number of iterations, and data
params = jnp.zeros (2)  # Example: 2 parameters initialized to zero
learning_rate = 0.01
num_iterations = 1000

# Example data: X (features) and y (targets)
X = jnp.array ([[1, 2], [2, 3], [3, 4], [4, 5]])
y = jnp.array ([3, 5, 7, 9])
data = (X, y)

# Train the model
optimized_params, loss_values = train_model (loss_fn, params, learning_rate, num_iterations, data)
plot_loss (loss_values)
