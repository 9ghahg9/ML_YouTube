# Numpy(II)
## dot prodcut
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[-1, -2], [-3, -4], [-5, -6]])
C = np.dot(A, B)
print("A.shape:", A.shape, "B.shape:", B.shape)
print("C.shape:", C.shape)
print(C)


## Numpy broadcast: It allows operations even between two matrices of different shapes. 
A = np.array([[1, 2], [3, 4]])
b = 5
print(A + b)

C = np.array([[1, 2], [3, 4]])
D = np.array([4, 5])
print(C + D)


## Transpose
A = np.array([[1, 2], [3, 4], [5, 6]])
B = A.T
print("A.shape:", A.shape, "B.shape:", B.shape)
print(A)
print(B)

C = np.array([1, 2, 3, 4, 5])   # 1D array(== vector)
D = C.T

E = C.reshape(1, 5)             # Make C a 1x5 matrix. Not a vector.
F = E.T
print("C.shape:", C.shape, "D.shape:", D.shape)
print("E.shape:", E.shape, "F.shape:", F.shape)
print(F)


## Indexing / Slicing
A = np.array([10, 20, 30, 40, 50, 60]).reshape(3, 2)
print("A.shape:", A.shape)
print(A)

print("A[0, 0]:", A[0, 0], "A[0][0]:", A[0][0])
print("A[2, 1]:", A[2, 1], "A[2][1]:", A[2][1])

print("A[0:-1, 1:2] ==", A[0:-1, 1:2])

print("A[:, 0] ==", A[:, 0])
print("A[:, :] ==", A[:, :])


## Iterator: It is used to access all elements of a matrix, 
##           aside from explicit(명시적) indexing or slicing.
import numpy as np
A = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
print(A, "\n")
print("A.shape:", A.shape, "\n")

it = np.nditer(A, flags=['multi_index'], op_flags=['readwrite'])

while not it.finished:
    idx = it.multi_index
    print("current value => ", A[idx])
    it.iternext()
