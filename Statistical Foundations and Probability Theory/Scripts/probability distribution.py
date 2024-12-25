import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, norm, uniform, poisson

# 1. Bernoulli Distribution
p = 0.6  # Probability of success
bernoulli_trials = bernoulli.rvs(p, size=10)
print("Bernoulli trials (success=1, failure=0):", bernoulli_trials)

# 2. Binomial Distribution
n, p = 10, 0.6  # 10 trials, probability of success = 0.6
binomial_outcomes = binom.rvs(n, p, size=1000)
plt.hist(binomial_outcomes, bins=np.arange(-0.5, n+1.5, 1), density=True, edgecolor='black')
plt.title("Binomial Distribution (n=10, p=0.6)")
plt.xlabel("Number of successes")
plt.ylabel("Probability")
plt.show()

# 3. Normal Distribution
mu, sigma = 0, 1  # Mean = 0, Standard Deviation = 1
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf, label=f"Normal(0,1)")
plt.title("Normal Distribution (μ=0, σ=1)")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()

# 4. Uniform Distribution
a, b = 0, 1  # Range [0, 1]
x = np.linspace(a, b, 1000)
pdf = uniform.pdf(x, a, b-a)
plt.plot(x, pdf, label="Uniform(0,1)")
plt.title("Uniform Distribution (0,1)")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()

# 5. Poisson Distribution
lambda_ = 3  # Average rate of occurrence
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_)
plt.bar(x, pmf, color='blue', alpha=0.7, edgecolor='black')
plt.title("Poisson Distribution (λ=3)")
plt.xlabel("Number of events")
plt.ylabel("Probability")
plt.show()
