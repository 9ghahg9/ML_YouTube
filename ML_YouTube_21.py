# Deep Learning(III)
## XOR probelm
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
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
class LogicGate:
    def __init__(self, gate_name, xdata, tdata):
        self.name = gate_name
        self.__xdata = xdata.reshape(4, 2)  # 'Hatch' processing matrix for 4 input data x1, x2
        self.__tdata = tdata.reshape(4, 1)  # The calculation value matrix for each of 4 input data x1, x2
        
        #$ 2-layer hidden layer unit: Assume 6 units, initialize weight W2, bias b2
        self.__W2 = np.random.rand(2, 6)    # weight, 2 X 6 matrix
        self.__b2 = np.random.rand(6)
        
        #$ 3-layer hidden layer unit: Assume 1 units, initialize weight W3, bias b3
        self.__W3 = np.random.rand(6, 1)
        self.__b3 = np.random.rand(1)

        self.__learning_rate = 1e-2



    def __feed_forward(self):       # Calculate the loss func value througt feedforward.
        delta = 1e-7

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2  # linear regression value of the hidden layer.
        a2 = sigmoid(z2)                                  # Output of the hidden layer.

        z3 = np.dot(a2, self.__W3) + self.__b3            # linear regression value of the output layer.
        a3 = sigmoid(z3)                                  # Output of the output layer.
        y = a3 = sigmoid(z3)
        return -np.sum(self.__tdata * np.log(y + delta) + 
                       (1 - self.__tdata) * np.log((1 - y) + delta)) 

    def __loss_val(self):           # Calculate of the loss func value for external output.
        delta = 1e-7

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2  # linear regression value of the hidden layer.
        a2 = sigmoid(z2)                                  # Output of the hidden layer.

        z3 = np.dot(a2, self.__W3) + self.__b3            # linear regression value of the output layer.
        a3 = sigmoid(z3)                                  # Output of the output layer.
        y = a3 = sigmoid(z3)
        return -np.sum(self.__tdata * np.log(y + delta) + 
                       (1 - self.__tdata) * np.log((1 - y) + delta)) 
    

    ## "No-error_val function"


    def train(self):
        f = lambda x: self.__feed_forward()                 # loss_func   -> feed_forward
        print("Initial loss value =", self.__loss_val())    # error vaule -> loss value
        for step in range(10001):

            self.__W2 -= self.__learning_rate * numerical_derivation(f, self.__W2)
            self.__b2 -= self.__learning_rate * numerical_derivation(f, self.__b2)
            self.__W3 -= self.__learning_rate * numerical_derivation(f, self.__W3)
            self.__b3 -= self.__learning_rate * numerical_derivation(f, self.__b3)
            
            if (step % 400 == 0):
                print("step =", step, "loss value =", self.__loss_val())


    def predict(self, input_data):
        
        z2 = np.dot(input_data, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.__W3) + self.__b3
        y = a3 = sigmoid(z3)

        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

#! 1. AND_Gate
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([0, 0, 0, 1])

AND_obj = LogicGate("AND_GATE", xdata, tdata)

#!# AND Data training
AND_obj.train()

#!# AND Data prediction
print(AND_obj.name, "\n")

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, "=", logical_val, '\n')


#! 2. OR_Gate
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([0, 1, 1, 1])

OR_obj = LogicGate("OR_GATE", xdata, tdata)

#!# OR Data training
OR_obj.train()

#!# OR Data prediction
print(OR_obj.name, "\n")

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print(input_data, "=", logical_val, '\n')


#! 3. NAND_Gate
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([1, 1, 1, 0])

NAND_obj = LogicGate("NAND_GATE", xdata, tdata)

#!# NAND Data training
NAND_obj.train()

#!# NAND Data prediction
print(NAND_obj.name, "\n")

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print(input_data, "=", logical_val, '\n')


#! 4. XOR_Gate <- Great!
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([0, 1, 1, 0])

XOR_obj = LogicGate("XOR_GATE", xdata, tdata)
XOR_obj.train()

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for input_data in test_data:
    (sigmoid_val, Logical_val) = XOR_obj.predict(input_data)
    print(input_data, "=", Logical_val)
