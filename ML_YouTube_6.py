# Numpy(I)
## Overview
import numpy
A = numpy.array([1, 2])
print("A ==", A, ", type ==", type(A))

import numpy as np
A = np.array([3, 4])
print("A ==", A, ", type ==", type(A))

from numpy import exp
result = exp(1)
print("result ==", result, ", type ==", type(result))

from numpy import *
result = exp(1) + log(1.7) + sqrt(2)
print("result ==", result, ", type ==", type(result))


## Numpy is an essential library frequently used in machine learning code devlopment for 
## represeting and performing operations on vectors and matrices and more.

## numpy vs list
### Represnet a matrix as a list.
A = [[1, 0], [0, 1]]
B = [[1, 1], [1, 1]]
A + B     # List operations instead of(~이 아니라) matrix operations.

### Represnet a matrix as a numpy array.
A = np.array([[1, 0], [0, 1]])
B = np.array([[1, 1], [1, 1]])
A + B     # Matrix operations


## numpy vector (1D array)
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print("A ==", A, ", B ==", B)
print("A.shape ==", A.shape, ", B.shape ==", B.shape)
print("A.ndim ==", A.ndim, ", B.ndim ==", B.ndim)

### Vector operations
print("A + B ==", A + B)
print("A - B ==", A - B)
print("A * B ==", A * B)
print("A / B ==", A / B)


## numpy matrix (matrix)
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[-1, -2, -3], [-4, -5, -6]])
print("A.shape ==", A.shape, ", B.shape ==", B.shape)
print("A.ndim ==", A.ndim, ", B.ndim ==", B.ndim)

### Matrix reshaping
C = np.array([1, 2, 3])
print("C.shape ==", C.shape)
C = C.reshape(1, 3)    # Reshape C to a 1x3 matrix.
print("C.shape ==", C.shape)
