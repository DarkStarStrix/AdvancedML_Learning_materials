import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
X = np.random.rand(100, 2)
true_beta = np.array([2, -1])
true_sigma = 0.5
y = X.dot(true_beta) + np.random.normal(0, true_sigma, 100)

# Add a column of ones to X for the intercept term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Define prior parameters
beta_prior_mean = np.zeros(X.shape[1])
beta_prior_cov = np.eye(X.shape[1]) * 10
sigma_prior_shape = 1
sigma_prior_scale = 1

# Compute posterior parameters
XtX = X.T.dot(X)
XtX_inv = np.linalg.inv(XtX + np.linalg.inv(beta_prior_cov))
beta_post_mean = XtX_inv.dot(X.T.dot(y) + np.linalg.inv(beta_prior_cov).dot(beta_prior_mean))
beta_post_cov = XtX_inv
sigma_post_shape = sigma_prior_shape + len(y) / 2
sigma_post_scale = sigma_prior_scale + 0.5 * (y.dot(y) + beta_prior_mean.dot(np.linalg.inv(beta_prior_cov).dot(beta_prior_mean)) - beta_post_mean.dot(np.linalg.inv(beta_post_cov).dot(beta_post_mean)))

# Sample from the posterior
num_samples = 10000
beta_samples = np.random.multivariate_normal(beta_post_mean, beta_post_cov, num_samples)
sigma_samples = np.sqrt(1 / np.random.gamma(sigma_post_shape, 1 / sigma_post_scale, num_samples))

# Plot the posterior distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(beta_samples[:, 1], bins=30, density=True, alpha=0.6, color='g')
plt.title('Posterior of beta[0]')
plt.subplot(1, 2, 2)
plt.hist(beta_samples[:, 2], bins=30, density=True, alpha=0.6, color='b')
plt.title('Posterior of beta[1]')
plt.show()

# Predict new data
X_new = np.random.rand(10, 2)
X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new))
true_beta_with_intercept = np.array([0, 2, -1])  # Include intercept term
y_new = X_new.dot(true_beta_with_intercept) + np.random.normal(0, true_sigma, 10)
y_pred_samples = X_new.dot(beta_samples.T)

# Plot the predictions
plt.scatter(y_new, y_pred_samples.mean(axis=1))
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Predicted vs. True values")
plt.show()
