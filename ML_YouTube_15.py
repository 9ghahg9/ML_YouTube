# Linear Regression
#@ ex1.
import numpy as np

loaded_text = np.loadtxt("./data-15-input_answer.csv", delimiter=',', dtype=np.float32, 
                         skiprows=1, encoding='utf-8')
x_data = loaded_text[:, [0]]
t_data = loaded_text[:, [1]]

W = np.random.rand(1, 1)
b = np.random.rand(1)

print("W.shape ==", W.shape, "b.shape ==", b.shape, "W ==", W, ", b ==", b)

def loss_function(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y)**2) / len(x)

def numerical_1st_derivative(f, x):
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
    y = np.dot(x, W) + b
    return np.sum((t - y)**2) / len(x)

def predict(x):
    y = np.dot(x, W) + b
    return y

learning_rate = 1e-2

f = lambda x: loss_function(x_data, t_data)

print("Initial error value =", error_val(x_data, t_data), "Initial W =", W, "\n", ", b =", b)

for step in range(10001):
    W -= learning_rate * numerical_1st_derivative(f, W)
    b -= learning_rate * numerical_1st_derivative(f, b)
    if (step % 400 == 0):
        print("step =", step, "error value =", error_val(x_data, t_data), 
              "W =", W, " , b =", b)
result = predict(43)
print("result =", result)


#@ ex2.
import numpy as np

nloaded_text = np.loadtxt("./data-15-test_score.csv", 
                          delimiter=',', dtype=np.float32,
                          skiprows=1, encoding='utf-8')
x_ndata = nloaded_text[:, 0:-1]
t_ndata = nloaded_text[:, [-1]]

W = np.random.rand(3, 1)
b = np.random.rand(1)

print("W.shape ==", W.shape, "b.shape ==", b.shape, "W ==", W, ", b ==", b)

def loss_func(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y)**2) / len(x)

def numerical_derivative(f, x):
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

def nerror_val(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y)**2) / len(x)
def npredict(x):
    y = np.dot(x, W) + b
    return y

print("Initial error value =", nerror_val(x_ndata, t_ndata), 
      "Initial W =", W, "\n", " , b =", b)

nlearning_rate = 1e-5

f = lambda X: loss_func(x_ndata, t_ndata)

for step in range(10001):
    W -= nlearning_rate * numerical_derivative(f, W)
    b -= nlearning_rate * numerical_derivative(f, b)
    if (step % 400 == 0):
        print("step =", step, "error value =", nerror_val(x_ndata, t_ndata), 
              "W =", W, ", b =", b)
        
nresult = npredict([100, 98, 81])
print("nresult =", nresult)
