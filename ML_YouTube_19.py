# Deep Learning(I)
## Neural Network: It refers to receiving input signals from previous neurons 
##                the generating another signal.
##
##              => However, instead of producing output proportional 
##                to the input (y = Ex), it generates an output signal only 
##                when the total sum of the input values reaches a certain threshold.
##
##              => And the function that defines this threshold is 
##                called the activation function.


##                 So, to apply the operating principles of neurons 
##                in the nervous system to machine learning,
##
##             1. after multiplying the input signals by the weights 
##                and adding an appropriate bias,
##             2. the value is passed as input to the activation function, using 
##                the sigmoid function as an example, where a threshold greater than 
##                0.5 outputs 1, and otherwise 0, which is then passed to the next neuron.
##             ==> Building multi-variable, Logistic Regression systems.


## Deep Learning: It is a field of machine learning that builds an input layer, 
##               one or more hidden layers, and an output layer based on a neural network 
##               structure where nodes(neurons) are interconnected, 
##               and learns the weights of each node(neuron) based on the error 
##               in the output layer.

## A weight W21 => It is that strengthen or weaken the signals transmitted from Node1 of 
##                a specific layer to Node2 in the next layer. (W'21' => ( 1 -> 2 ))


#!!!: 정리! 입력층과 출력층만 있는 경우는 2차원 공간.
#!!!  중간에 은닉층이 들어가면, 공간에 왜곡이 발생, 즉 2차원 공간에서 해결하지 못했던 
#!!!  데이터 분류를 공간 왜곡으로서 해결하는 방식이다.
#!!!
#!!!  그렇기에, 비선형이라는 것이 2차 함수, 3차 함수 이런게 아니라, 공간에 곡률을 줘서 
#!!!  공간에 왜곡이 생겨 직선으로도, 전에 2차원 공간에선 할 수 없었던 데이터 분류를 
#!!!  가능하게 해주도록 은닉층이 만들어 준다.
#!!!
#!!!  그렇기에 굳이 곡률을 알 필요가 없다. W와 b는 numerical_derivative 함수로 자동으로 
#!!!  t에 가까워지려 하기 때문이다.
