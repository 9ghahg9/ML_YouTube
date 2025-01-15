# Function => def function_name(input1, input2, ...):
## ex. def
def sum(x, y):
    s = x + y
    return s
result = sum(10, 20)
print(result)

def multi_ret_func(x):
    return x+1, x+2, x+3    # return tuple and (x+1, x+2, x+3)
x = 100
y1, y2, y3 = multi_ret_func(x)
print(y1, y2, y3)


## Default Parameter
def print_name(name, count=2):  # count=2 is default parameter. 
                                # So, print_name is initialized with 2.
    for i in range(count):
        print("name ==", name)
print_name("DAVE")    # print_name("DAVE", 5) => output 5 times, not 2 times.

## Mutable / Immutable parameter
##  => Mutable: list, dict, ...            => The value of the parameter 'can' be changed.
##  => Immutable: int, str, tuple, ...     => The value of the parameter 'cannot' be changed.
##                                            So, save the immutable parameter
def mutable_immutable_func(int_x, input_list):
    int_x += 1
    input_list.append(100)
    print("int_x ==", int_x, "input_list ==", input_list)


## Lambda Function => Function name = lambda input1, input2, ... : return value
f = lambda x: x + 100
for i in range(3):
    print(f(i))

#@ ex. lambda
def print_hello():
    print("Hello Python!")
def test_lambda(s, t):
    print("input 1 ==", s, "input 2 ==", t)
s = 100
t = 200
fx = lambda x, y: test_lambda(s, t)
fy = lambda x, y: print_hello()
fx(500, 1000)
fy(300, 600)
