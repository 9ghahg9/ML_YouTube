# Logistic Regression - Classification(I)
## Logistic Regression Algorithm
## 1. Find the optimal line that represents the characteristics and distribution 
##    of the training data.
## 2. It is an algorithm that classifies data as above(1) or below(0) based on the line.
## ==> (x, t) => Regression => Classification => True(1) or False(0)


## Classification - sigmoid funciton
## (x, t) => Regression =>(z = Wx + b) 
##           Classification =>(y = sigmoid(z))
##           True(1) or False(0)

## The output value y should have 1 or 0. So, sigmoid value > 0.5   => True(1),
##                                                          <= 0.5  => False(0)
## y = 1 / (1 + exp(-[Wx + b])), t = 0 or 1
## E(W, b) = -np.sum[i=1, n]( ti * log(yi) + (1 - ti) * log(1 - yi) )

## p(C=1|x) = y = sigmoid(Wx + b)
## p(C=0|x) = 1 - p(C=1|x) = 1 - y
## p(C=t|x) = y^t * (1 - y)^(1 - t)
## L(W, b)  = Π[i=1, n]P(C=ti|xi) = Π[i=1,n]yi^ti * (1 - y1)^(1 - ti)
## E(W, b)  = -log L(W, b)
##          = -sum[i=1, n]( ti * log(yi) + (1 - ti) * log(1 - yi) )
