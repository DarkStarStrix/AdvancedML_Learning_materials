import jax
import jax.numpy as jnp
import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Example: Comparing matrix multiplication performance

# NumPy
start_time = time.time ()
np_result = np.matmul (np.random.rand (2000, 2000), np.random.rand (2000, 2000))
np_time = time.time () - start_time

# JAX (without JIT)
start_time = time.time ()
jax_result = jnp.matmul (jax.random.normal (jax.random.PRNGKey (0), (2000, 2000)),
                         jax.random.normal (jax.random.PRNGKey (1), (2000, 2000)))
jax_time = time.time () - start_time


# JAX (with JIT)

@jax.jit
def jax_matmul(a, b):
    return jnp.matmul (a, b)


start_time = time.time ()
jax_jit_result = jax_matmul (jax.random.normal (jax.random.PRNGKey (0), (2000, 2000)),
                             jax.random.normal (jax.random.PRNGKey (1), (2000, 2000)))
jax_jit_time = time.time () - start_time

print (f"NumPy time: {np_time:.4f} seconds")
print (f"JAX time (without JIT): {jax_time:.4f} seconds")
print (f"JAX time (with JIT): {jax_jit_time:.4f} seconds")


# visualizing the results writes a function that compares the results of the three methods
def compare_results(np_result, jax_result, jax_jit_result):
    print ("NumPy result:")
    print (np_result)
    print ("\nJAX result (without JIT):")
    print (jax_result)
    print ("\nJAX result (with JIT):")
    print (jax_jit_result)


compare_results (np_result, jax_result, jax_jit_result)


# Plotting the results
def plot_results(np_time, jax_time, jax_jit_time):
    labels = ['NumPy', 'JAX (without JIT)', 'JAX (with JIT)']
    times = [np_time, jax_time, jax_jit_time]

    plt.bar (labels, times)
    plt.ylabel ('Time (seconds)')
    plt.title ('Matrix Multiplication Performance Comparison')
    plt.show ()


plot_results (np_time, jax_time, jax_jit_time)
