# Numpy(III)
## Concatenate: It is used to add rows or columns to a matrix.
##              => It is a programming technique used in implementing regression code in 
##                 machine learning to treat weights and biases as a single matrix 
##                 instead of separating them.
import numpy as np
A = np.array([[10, 20, 30], [40, 50, 60]])
print(A.shape)
B = np.array([[70, 80, 90]]).reshape(1, 3)

row_add = np.array([70, 80, 90]).reshape(1, 3)
column_add = np.array([1000, 2000]).reshape(2, 1)

B = np.concatenate((A, row_add), axis=0)     # axis=0 => row-wise
print(B)
C = np.concatenate((A, column_add), axis=1)  # axis=1 => column-wise
print(C)


## Useful functions(I)
### loadtxt('file_name', delimiter=''(,=> 0,1,2, ;=> 0;1;2), 
###         dtype=type(int, float, str), encoding='utf-8')
loaded_data = np.loadtxt("./data-08.csv", delimiter=',', dtype=np.float32, encoding='utf-8')
x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print("x_data.ndim =", x_data.ndim, ", x_data.shape =", x_data.shape)
print("t_data.ndim =", t_data.ndim, ", t_data.shape =", t_data.shape)


## Useful functions(II)
### 0~1 random number
random_number1 = np.random.rand(3)
random_number2 = np.random.rand(1, 3)
random_number3 = np.random.rand(3, 1)

print("random_number1 =", random_number1, " , random_number1.shape =", random_number1.shape)
print("random_number2 =", random_number2, " , random_number2.shape =", random_number2.shape)
print("random_number3 =", random_number3, " , random_number3.shape =", random_number3.shape)

###
X = np.array([2, 4, 6, 8])
print("np.sum(X) =", np.sum(X))  # sum()
print("np.exp(X) =", np.exp(X))  # exp()
print("np.log(X) =", np.log(X))  # log()


## Useful functions(III)
###
X = np.array([2, 4, 6, 8])
print("np.max(X) =", np.max(X))  # max()
print("np.min(X) =", np.min(X))  # min()
print("np.argmax(X) =", np.argmax(X))  # argmax()
print("np.argmin(X) =", np.argmin(X))  # argmin()

#@ ex. max, min, argmax, argmin
X = np.array([[2, 4, 6], [1, 2, 3], [0, 5, 8]])
print("np.max(X) =", np.max(X, axis=0))        # max(), Column-wise
print("np.max(X) =", np.max(X, axis=1))        # max(), Row-wise

print("np.min(X) =", np.min(X, axis=0))        # min(), Column-wise
print("np.min(X) =", np.min(X, axis=1))        # min(), Row-wise

print("np.argmax(X) =", np.argmax(X, axis=0))  # argmax(), Column-wise
print("np.argmax(X) =", np.argmax(X, axis=1))  # argmax(), Row-wise

print("np.argmin(X) =", np.argmin(X, axis=0))  # argmin(), Column-wise
print("np.argmin(X) =", np.argmin(X, axis=1))  # argmin(), Row-wise

###
A = np.ones([3, 3])
print("A.shape =", A.shape, "\nA ==", A)
B = np.zeros([3, 2])
print("B.shape =", B.shape, "\nB ==", B)


## matplotlib / scatter plot
#@ ex. scatter plot
import matplotlib.pyplot as plt
import numpy as np

x_data = np.random.rand(100)
y_data = np.random.rand(100)

plt.title('Scatter plot')
plt.grid()
plt.scatter(x_data, y_data, color='b', marker='o')
plt.show()

#@ ex.1 line plot
import matplotlib.pyplot as plt

x_data = [x for x in range(-5, 5)]
y_data = [y*y for y in range(-5, 5)]

plt.title('Line plot')
plt.grid()
plt.plot(x_data, y_data, color='b')
plt.show()

#@ ex.2 line plot
import matplotlib.pyplot as plt

x_data = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_data = [-8, -13, 0, 3, 6, -1, -5, -7, 1, 8, 7, 12, 13]

plt.title('Line plot')
plt.grid()
plt.plot(x_data, y_data, color='b')
plt.show()
