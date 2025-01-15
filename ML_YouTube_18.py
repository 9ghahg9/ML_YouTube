# Logic Gate
## Prove Logic Gate Class: AND, OR, NAND, XOR, ...
## =>
##      1. External Function
##      def sigmoid(x):                                     # The function to output 0 or 1
##      def numerical_derivative(f , x)

##      2. LogicGate class
##      class LogicGate:
##          def __init__(self, gate_name, xdata, tdata):    # Initialize xdata, tdata, W, b
##          def loss_func(self):            # The loss function, cross-entropy.
##          def error_val(self):            # Calculate the loss function value.
##          def train(self):                # The method to find the minimum value of the 
##                                            loss function using numerical differentiation.
##          def predict(self, xdata):       # The method to predict future value.

##      3. Usage
##      xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data generation.
##      tdata = np.array([0 or 1, 0 or 1, 0 or 1, 0 or 1])  # Correct answer generation.

##      {LogicGate_class}_obj = LogicGate("{gate_name}", xdata, tdata)
##      {LogicGate_class}_obj.train()

##      {LogicGate_class}_obj.predict(...)


## Implementation (구현)
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
        
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)
        
        self.__W = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        self.__learning_rate = 1e-2
    
    def __loss_func(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        return -np.sum((self.__tdata * np.log(y + delta) + 
                        (1 - self.__tdata) * np.log((1 - y) + delta)))
    
    def __error_val(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        return -np.sum((self.__tdata * np.log(y + delta) + 
                        (1 - self.__tdata) * np.log((1 - y) + delta)))
    
    def train(self):
        f = lambda x: self.__loss_func()

        print("Initial error value =", self.__error_val())
        
        for step in range(10001):
            self.__W -= self.__learning_rate * numerical_derivation(f, self.__W)
            self.__b -= self.__learning_rate * numerical_derivation(f, self.__b)
            if (step % 400 == 0):
                print("step =", step, "Initial error value =", self.__error_val())

    def predict(self, input_data):
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)
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


#! 4. XOR_Gate?     => An error occurs. == t = [0, 1, 1, 0] != [0, 0, 0, 1]
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([0, 1, 1, 0])

XOR_obj = LogicGate("XOR_GATE?", xdata, tdata)

#!# XOR Data training
XOR_obj.train()

#!# XOR Data prediction
print(XOR_obj.name, "\n")

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for input_data in test_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(input_data)
    print(input_data, "=", logical_val, '\n')


#! Correct 4. XOR_Gate      => The core idea of deep learning based on neural networks.
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

s1 = []     # Output NAND
s2 = []     # Output OR

new_input_data = []
final_output = []

for index in range(len(input_data)):
    s1 = NAND_obj.predict(input_data[index])
    s2 = OR_obj.predict(input_data[index])

    new_input_data.append(s1[-1])
    new_input_data.append(s2[-1])

    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))

    final_output.append(logical_val)
    new_input_data = []

print('\n')
print("XOR_GATE")

for index in range(len(input_data)):
    print(input_data[index], "=", final_output[index], end=' ')
    print("\n")
