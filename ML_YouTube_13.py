# Linear Regression(I)
## step 1. Analyze training data
## : The learning data shows a tendency for the output(y), the test score, to increase 
## proportionally with the input(x), the study time.
## So, input(x), output(y) => y = Wx + b.

## step 2. Find W and b
## : The concept of learning is finding the weight W(slope) and the bias b(y-intercept).


## error, weight W, bias b
## If y = Wx + b and t is correct answers, error = t - y = t - (Wx + b).
## Therefore, if error values are minimized, predicted values increase.
## <=> We should need to find weight W and the bias b values thah can accurately predict 
##     future values.


## Loss[cost] Function: It's expressed as formula by summing up the differences between 
##                      the correct answer(t) in the training data and the calculeated 
##                      values y for the inputs.
## <=> (t - y)^2 = (t - [Wx + b])^2. # Errors can be +, -, zero, and squaring the errors 
##                                     ensures(보장한다) they are always positive.

## loss function = ( (t1 - y1)^2 + (t2 - y2)^2 + ... + (ti - yi)^2 ) / n
##               = ( (t1 - [Wx1 + b])^2 + (t2 - [Wx2 + b])^2 + 
##                 ... + (ti - [Wxi + b])^2 ) / n
##               = 1/n * sum[i=1, n](ti - [Wi + b])^2
##               = 1/n * sum[i=1, n](ti - yi)^2
##               = E(W, b)

## The ultimate goal of a (linear) regression model is to find (W, b) such that 
## the loss function has a minimum value.
