# MNIST: It is a dataset consisiting handwritten numbers(cursive digits), 
#       Just as we print 'Hello, World' when we learn new programming language, 
#       MNIST is like the 'Hello, World' that is essential when learn deep learning.

import numpy as np

## It exist 60,000 datas. 1 data seperates 785 number as ',', 
## and it consist of one number representing the consist correct answer and 
## 784 numbers representing the handwritten image.
training_data = np.loadtxt(r"C:\Users\skygr\Desktop\mnist_train.csv", delimiter=',', dtype=np.float32, encoding='utf-8')

## It exist 10,000 datas. It's consist of 785 numbers concluded test data or 
## correct(label).
test_data = np.loadtxt(r"C:\Users\skygr\Desktop\mnist_test.csv", delimiter=',', dtype=np.float32, encoding='utf-8')

print("training_data.shape ==", training_data.shape, "test_data.shape ==", test_data.shape)


## Recode, 1 row.
### 1. one recode = 785 columns.
### 2. ond column has a correct answer.
### 3. From the 2nd to the last column, there are 784 consecutive numerical values 
###    representing the colors of the image corresponding to the correct answer.
#### 3.- 'colors of the image': Closer to 0 indicates black, 
####                            while closer to 1 indicates white.
#import matplotlib.pyplot as plt
#img = training_data[0][1:].reshape(28, 28)    # 28 X 28 = 784
#plt.imshow(img, cmap='gray')
#plt.show()


## one-hot encoding
#@ ex1.
### Input layer:  1~784
### Hidden layer: node == 100; Arbitrarily decide. => W(2) = (784 X 100)
### Output layer: node == 10;  Because correct answer is one of 0 ~ 10.
###                            => Print the index with the largest value 
###                               among 0 to 9 as the correct answer.(one-hot encoding)
### 
### 1. External Function
### def sigmoid(x):
### def numerical_derivative(f, x):
###
### 2. NeuralNetwork class
### class NeuralNetwork:
###   def __init__(self, input_datas, hidden_datas, output_datas)   # Initialize weight, bias, learning rate
###   def feed_forward(self)    # Calculate the loss func value using feedforward.
###   def loss_val(self)        # Calculate the loss func value (for external output).
###   def train(self, training_data) # Update weights/biases using numerical_derivative.
###   def predict(self, input_data)  # Predict the future value about input datas.
###   def accuracy(self, test_data)  # Measure the deep learning architecture accuracy 
###                                    based on neuralnetwork.
###
### 3. Usage
### nn =  NeuralNetwork(784, 100, 10)
### for step in range(30001):               # Training is conducted using only 50%, or 
###                                           30,000, of the 60,000 training data.
###   index = np.random.randint(0, 59999)
###   nn.train(training, data[index])

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
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
class NeuralNetwork:
  def __init__(self, input_nodes, hidden_nodes, output_nodes):  # Initialize weights, biases, a learning rate
    self.__input_nodes = input_nodes    # input_nodes  = 784
    self.__hidden_nodes = hidden_nodes  # hidden_nodes = 100
    self.__output_nodes = output_nodes  # output_nodes = 10
    
    self.__W2 = np.random.rand(self.__input_nodes, self.__hidden_nodes)
    self.__b2 = np.random.rand(self.__hidden_nodes)
    
    self.__W3 = np.random.rand(self.__hidden_nodes, self.__output_nodes)
    self.__b3 = np.random.rand(self.__output_nodes)
    
    self.__learning_rate = 1e-4
    
  def __feed_forward(self):
    delta = 1e-7
    z2 = np.dot(self.__input_nodes, self.__W2) + self.__b2
    y2 = sigmoid(z2)
    
    z3 = np.dot(y2 , self.__W3) + self.__b3
    a3 = y3 = sigmoid(z3)
    
    return -np.sum((self.__target_data * np.log(a3 + delta) + 
                    (1 - self.__target_data) * np.log((1 - a3) + delta)))
    
  def loss_val(self):
    delta = 1e-7
    z2 = np.dot(self.__input_nodes, self.__W2) + self.__b2
    y2 = sigmoid(z2)
    
    z3 = np.dot(y2 , self.__W3) + self.__b3
    a3 = y3 = sigmoid(z3)
    
    return -np.sum((self.__target_data * np.log(a3 + delta) + 
                    (1 - self.__target_data) * np.log((1 - a3) + delta)))
    
  def train(self, training_data):
    
    # one-hot encoding
    self.__target_data = np.zeros(self.__output_nodes) + 0.01
    self.__target_data[int(training_data[0])] = 0.99    # 1-column is a correct answer, another are color datas.
    self.__input_nodes = (training_data[1:] / 255.0 * 0.99) + 0.01
    
    f = lambda x: self.__feed_forward()
    
    self.__W2 -= self.__learning_rate * numerical_derivative(f, self.__W2)
    self.__b2 -= self.__learning_rate * numerical_derivative(f, self.__b2)
    self.__W3 -= self.__learning_rate * numerical_derivative(f, self.__W3)
    self.__b3 -= self.__learning_rate * numerical_derivative(f, self.__b3)
    
  def predict(self, input_data):
    z2 = np.dot(input_data, self.__W2) + self.__b2
    y2 = sigmoid(z2)
    z3 = np.dot(y2, self.__W3) + self.__b3
    a3 = y3 = sigmoid(z3)
    
    predicted_num = np.argmax(a3)
    
    return predicted_num
  
  def accuracy(self, test_data):
    matched_list = []
    not_matched_list = []
    
    for index in range(len(test_data)):
      label = int(test_data[index, 0])
      
      data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
      
      predicted_num = self.predict(data)
      
      if label == predicted_num:
        matched_list.append(index)
      else:
        not_matched_list.append(index)
      
    print("Current Accuracy =", 100 * (len(matched_list) / (len(test_data))), " %")
    
    return matched_list, not_matched_list


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

for step in range(30001):
  index = np.random.randint(0, len(training_data)-1)
  nn.train(training_data[index])
  if step % 400 == 0:
    print("step =", step, ", loss_val =", nn.loss_val())

nn.accuracy(test_data)

## Iniuitive but time-consuming for derivative calculations.
## So, Solution: Back Propagation
