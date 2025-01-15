# Machine Learning Numerical derivatives(II)
## Numerical 1st Derivative
def numerical_1st_derivative(f, x):
    delta_x = 1e-4
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)

#@ ex1. f(x) = x^2, x = 3
def my_func1(x):
    return x**2
result1 = numerical_1st_derivative(my_func1, 3)
print("result1 ==", result1)

#@ ex2. f(x) = 3xe^x, x = 2
import numpy as np
def my_func2(x):
    return 3 * x * np.exp(x)
result2 = numerical_1st_derivative(my_func2, 2)
print("result2 ==", result2)


## Numerical nth Derivative
def numerical_nth_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x)
        if np.isscalar(fx1) and np.isscalar(fx2):
            grad[idx] = (fx1 - fx2) / (2 * delta_x)
        else:
            grad[idx] = (fx1[idx] - fx2[idx]) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad

