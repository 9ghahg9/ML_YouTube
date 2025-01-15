# Date Type
## list
##  => 0, 1, 2, ... The index starts from the beginning of the list.
##  => -1, -2, -3, ... The index starts from the end of the list.
a = [10, 20, 30, 40, 50]
print("a[0] ==", a[0], "a[2] ==", a[2], "a[4] ==", a[4])
print("a[-1] ==", a[-1], "a[-2] ==", a[-2], "a[-5] ==", a[-5])

b = [10, 20, "Hello", [True, 3.14]]
print("b[0] ==", b[0], "b[2] ==", b[2], "b[3] ==", b[3])
print("b[-1] ==", b[-1], "b[-2] ==", b[-2], "b[-4] ==", b[-4])

print("b[3][0] ==", b[3][0], "b[3][1] ==", b[3][1])
print("b[-1][-1] ==", b[-1][-1], "b[-1][-2] ==", b[-1][-2])

c = []
c.append(100), c.append(200), c.append(300)
print("c ==", c)

print("a[0:2] ==", a[0:2], "a[1:] ==", a[1:])       # It's called slicing.
print("a[:3] ==", a[:3], "a[:-2] ==", a[:-2])
print("a[:] ==", a[:])


## Tuple
##  => The elements of the tuple cannot be changed.
a = (10, 20, 30, 40, 50)
print("a[0] ==", a[0], "a[-2] ==", a[-2], "a[:] ==", a[:])
print("a[0:2] ==", a[0:2], "a[1:] ==", a[1:])

#a[0] = 100     # ex. TypeError: 'tuple' object does not support item assignment


## Dictionary
scroe = {"KIM":90, "LEE":85, "JUN":95}
print("scroe['KIM'] ==", scroe['KIM'])

scroe['HAN'] = 100          # Dictionaries do not store data in the order it was entered, 
print("scroe ==", scroe)    # so caution is needed.

print("score key ==", scroe.keys())
print("score value ==", scroe.values())
print("score items ==", scroe.items())      # keys + values


## String
a = 'A73,CD'     # a[0] = A, a[5] = D
a[1]             # string 7. not int 7

a = a + ', EFG'
a                # print 'A73,CD, EFG'

b = a.split(',') # split string by ','
print(b)


## Useful Functions
a = [10, 20, 30, 40, 50]
b = (10, 20, 30, 40, 50)
c = {"KIM":90, "LEE":80}
d = 'Seoul, Korea'
e = [[100, 200], [300, 400], [500, 600]]

print("type(a) ==", type(a), "type(b) ==", type(b),                   # type()
      "type(c) ==", type(c), "type(d) ==", type(d), "type(e) ==", type(e))
print(len(a), len(b), len(c), len(d), len(e))                         # len()
import numpy as np
print(np.size(e))                                                     # np.size()

a = 'Hello'
b = {'KIM':90, 'LEE':80}

print(list(a), list(b.keys()), list(b.values()), list(b.items()))     # list()
str(3.14), str(100), str([1, 2, 3])                                   # str(), Only string
int('100'), int(3.14), float('3.14'), float(100)                      # int(), float()
