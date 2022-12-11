import numpy as np

# Aij is 1 if j links to i else 0 

matrix = np.array([
        [0,0,0,0,1,0],
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [1,1,0,0,1,0],
        [1,0,1,1,0,1],
        [1,0,0,0,0,0],
    ])

matrix = matrix / np.sum(matrix, axis=0, keepdims=True)

from main import powerMethod

eigvec, eigval = powerMethod(matrix)

print("largest rank", np.argmax(np.abs(eigvec)) + 1)

