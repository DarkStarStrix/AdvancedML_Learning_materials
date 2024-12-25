import torch
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# PyTorch
x = torch.tensor (2.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward ()
print (f"PyTorch Gradient: {x.grad}")

# JAX
x = jnp.array (2.0)


def f(x):
    return x ** 2 + 2 * x + 1


grad_f = jax.grad (f)
print (f"JAX Gradient: {grad_f (x)}")


# plot the gradients
def plot_gradients():
    x = jnp.linspace (-10, 10, 100)
    y = f (x)
    dy_dx = jax.vmap (grad_f) (x)  # Use vmap to apply grad_f to each element in x
    plt.figure (figsize=(10, 5))
    plt.plot (x, y, label='f(x)')
    plt.plot (x, dy_dx, label="f'(x)")
    plt.xlabel ('x')
    plt.ylabel ('y')
    plt.legend ()
    plt.show ()


plot_gradients ()
