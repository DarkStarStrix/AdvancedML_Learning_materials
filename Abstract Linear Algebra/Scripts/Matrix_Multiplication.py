import numpy as np

# Create two matrices
matrix_a = np.random.rand(1000, 1000)
matrix_b = np.random.rand(1000, 1000)

# Perform matrix multiplication
result = np.matmul(matrix_a, matrix_b)

print(result)
