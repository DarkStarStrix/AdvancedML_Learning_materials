import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt

# Set random seed for reproducibility
key = random.PRNGKey (42)


# Generate synthetic data
def generate_data(key, n_samples=300):
    key, subkey = random.split (key)
    means = jnp.array ([[0, 0], [5, 5], [0, 5]])
    covs = jnp.array ([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0.5], [0.5, 1]]])
    weights = jnp.array ([0.3, 0.4, 0.3])

    def sample_gaussian(key, mean, cov):
        return random.multivariate_normal (key, mean, cov)

    samples = []
    for i in range (n_samples):
        key, subkey = random.split (key)
        component = random.choice (subkey, 3, p=weights)
        key, subkey = random.split (key)
        sample = sample_gaussian (subkey, means [component], covs [component])
        samples.append (sample)

    return jnp.array (samples)


data = generate_data (key)

# Plot the synthetic data
plt.scatter (data [:, 0], data [:, 1], s=10)
plt.title ("Synthetic Data")
plt.show ()


# Define the GMM model
def log_prob_gaussian(x, mean, cov):
    d = x.shape [-1]
    cov_inv = jnp.linalg.inv (cov)
    diff = x - mean
    return -0.5 * (jnp.log (jnp.linalg.det (cov)) + diff @ cov_inv @ diff.T + d * jnp.log (2 * jnp.pi))


def e_step(data, means, covs, weights):
    def log_prob(x):
        return jnp.array (
            [log_prob_gaussian (x, means [k], covs [k]) + jnp.log (weights [k]) for k in range (len (weights))])

    log_probs = vmap (log_prob) (data)
    log_probs = log_probs - jnp.max (log_probs, axis=1, keepdims=True)
    probs = jnp.exp (log_probs)
    probs = probs / jnp.sum (probs, axis=1, keepdims=True)
    return probs


def m_step(data, probs):
    N_k = jnp.sum (probs, axis=0)
    weights = N_k / data.shape [0]
    means = jnp.dot (probs.T, data) / N_k [:, None]
    covs = jnp.array ([jnp.dot ((probs [:, k] [:, None] * (data - means [k])).T, data - means [k]) / N_k [k] for k in
                       range (len (weights))])
    return means, covs, weights


def gmm_em(data, n_components=3, n_iter=100):
    key = random.PRNGKey (42)
    key, subkey = random.split (key)
    means = random.normal (subkey, (n_components, data.shape [1]))
    covs = jnp.array ([jnp.eye (data.shape [1])] * n_components)
    weights = jnp.ones (n_components) / n_components

    for _ in range (n_iter):
        probs = e_step (data, means, covs, weights)
        means, covs, weights = m_step (data, probs)

    return means, covs, weights


# Fit the GMM model
means, covs, weights = gmm_em (data)

# Plot the results
plt.scatter (data [:, 0], data [:, 1], s=10, label="Data")
plt.scatter (means [:, 0], means [:, 1], s=100, c='red', label="GMM Means")
plt.title ("GMM Clustering")
plt.legend ()
plt.show ()
