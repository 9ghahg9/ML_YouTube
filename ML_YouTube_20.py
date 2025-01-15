# Deep Learning(II)
## Feed Forward - notation
### weight notation:            W21(2) => Node 1 of 1 layer -> Node 2 of 2 layer.
###                             (ab: node(b->a),  (): layer)
### bias notation:              b1(2)  => Node 1 of 2 layer.
###
### linear regression notation: z2(2)  => Node 2 of 2 layer.
###                             z2(2) = x1 W21(2) + x2 W22(2) + b2(2)
### node output notation :      a2(2)  => Node 2 of 2 layer.
###                             a2(2) = sigmoid(z2(2))

## Feed Forward - Operating mechanism
### Output input layer:         ( a1(1) = x1, a2(1) = x2 ) 
###                             == ( [a1(1), a2(1)] = [x1, x2] )
###
### Hidden layer linear regression value: 
###                             ( z1(2) = a1(1)W11(2) + a2(1)W12(2) + b1(2),
###                               z2(2) = a1(1)W21(2) + a2(1)W22(2) + b2(2) )
###                             == ( [z1(2), z2(2)] = 
###                                [a1(1), a2(1) * [[W11(2), W12(2)], [W21(2), W22(2)]] + 
###                                [b1(2), b2(2)] )
###
### Output hidden layer:        a1(2) = sigmoid(z1(2)), a2(2) = sigmoid(z2(2))
###
### Output layer linear regression value: 
###                             ( z1(3) = a1(2)W11(3) + a2(2)W12(3) + b1(3) )
###                             == ( [z1(3)] = 
###                                [a1(2), a2(2)] * [W11(3), W12(3)].T + [b1(3)] )
###
### Output output layer:        y = a1(3) = sigmoid(z1(3))
