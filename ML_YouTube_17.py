# Logistic Regression - Classification(II)
#@ ex1.
import numpy as np

loaded_text = np.loadtxt("./data-17-time_failpass.csv", delimiter=',', dtype=np.float32, 
                         skiprows=1, encoding='utf-8')
x_data = loaded_text[:, [0]]
t_data = loaded_text[:, [1]]

W = np.random.rand(1, 1)
b = np.random.rand(1)

print("W.shape ==", W.shape, "b.shape ==", b.shape, "W ==", W, ", b =", b)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def loss_func(x, t):
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))

def numerical_derivation(f, x):
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
        if np.isscalar(fx1):
            grad[idx] = (fx1 - fx2) / (2 * delta_x)
        else:
            grad[idx] = (fx1[idx] - fx2[idx]) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad
def error_val(x, t):
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))
def predict(x):
    z = np.dot(x, W) + b
    y = sigmoid(z)
    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result

learning_rate = 1e-2

f = lambda x: loss_func(x_data, t_data)

print("Initial error value =", error_val(x_data, t_data), 
      "Initial W =", W, '\n', ", b =", b)

for step in range(10001):
    W -= learning_rate * numerical_derivation(f, W)
    b -= learning_rate * numerical_derivation(f, b)
    if (step % 400 == 0):
        print("step =", step, "error value =", error_val(x_data, t_data), 
              "W =", W, ", b =", b)

(real_val, logical_val) = predict(3)
print(real_val, logical_val)

(real_val, logical_val) = predict(17)
print(real_val, logical_val)


#@ ex2.
import numpy as np

loaded_text = np.loadtxt("./data-17-review_prepare.csv", delimiter=',', dtype=np.float32, 
                         skiprows=1, encoding='utf-8')
x_data = loaded_text[:, 0:-1]
t_data = loaded_text[:, [-1]]
W = np.random.rand(2, 1)
b = np.random.rand(1)
print("W.shape ==", W.shape, "b.shape ==", b.shape, "W =", W, ", b =", b)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def loss_func(x, t):
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))
def numerical_derivation(f, x):
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
        if np.isscalar(fx1):
            grad[idx] = (fx1 - fx2) / (2 * delta_x)
        else:
            grad[idx] = (fx1[idx] - fx2[idx]) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad
def error_val(x, t):
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))
def predict(x):
    z = np.dot(x, W) + b
    y = sigmoid(z)
    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result

print("Initial error value =", error_val(x_data, t_data), 
      "Initial W =", W, ", b =", b)

learning_rate = 1e-2
f = lambda x: loss_func(x_data, t_data)

for step in range(10001):
    W -= learning_rate * numerical_derivation(f, W)
    b -= learning_rate * numerical_derivation(f, b)
    if (step % 400 == 0):
        print("step =", step, "error value =", error_val(x_data, t_data), 
              "W =", W, ", b =", b)

(real_val, logical_val) = predict([3, 17])
print(real_val, logical_val)

(real_val, logical_val) = predict([5, 8])
print(real_val, logical_val)

(real_val, logical_val) = predict([7, 21])
print(real_val, logical_val)

(real_val, logical_val) = predict([12, 0])
print(real_val, logical_val)
