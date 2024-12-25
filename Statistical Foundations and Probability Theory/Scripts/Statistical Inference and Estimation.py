import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Example data: Sample heights (in cm)
sample_heights = [170, 165, 180, 175, 169, 173, 172, 181, 168, 174]

# Step 1: Calculate the sample mean
sample_mean = np.mean(sample_heights)

# Step 2: Calculate the sample standard deviation
sample_std = np.std(sample_heights, ddof=1)  # Use ddof=1 for sample standard deviation

# Step 3: Determine the sample size
n = len(sample_heights)

# Step 4: Calculate the standard error of the mean
standard_error = sample_std / np.sqrt(n)

# Step 5: Determine the critical value for 95% confidence level
critical_value = stats.t.ppf(0.975, df=n-1)  # Two-tailed critical value for 95%

# Step 6: Calculate the margin of error and confidence interval
margin_of_error = critical_value * standard_error
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

# Visualization
x = np.linspace(165, 185, 1000)  # Range of possible heights for x-axis
y = stats.norm.pdf(x, sample_mean, sample_std)  # Normal distribution curve

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Normal Distribution Curve', color='blue')
plt.axvline(sample_mean, color='red', linestyle='--', label='Sample Mean')
plt.axvline(confidence_interval[0], color='green', linestyle='--', label='Lower Bound CI (95%)')
plt.axvline(confidence_interval[1], color='green', linestyle='--', label='Upper Bound CI (95%)')
plt.fill_betweenx(y, confidence_interval[0], confidence_interval[1], color='green', alpha=0.2, label='Confidence Interval')

plt.title('Confidence Interval for Sample Mean')
plt.xlabel('Height (cm)')
plt.ylabel('Density')
plt.legend()
plt.show()
