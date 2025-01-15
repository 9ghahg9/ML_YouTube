# If, For, While
## If
a = 1
if a > 0:
    print("a ==", a)
    print("a is a positive number.")
elif a == 0:
    print("a ==", a)
    print("a is zero.")
else:
    print("a ==", a)
    print("a is a negative number.")

## If condition in list, dict, ...
list_data = [10, 20, 30, 40, 50]
dict_data = {"key1":1, "key2":2}
if 45 in list_data:
    print("45 is in list_data.")
else:
    print("45 is not in list_data.")

if "key1" in dict_data:
    print("key1 is in dict_data.")
else:
    print("key1 is not in dict_data.")


## For
for data in list_data:
    print(data, " ", end="")    # end="" is no line breaks.
print()
for data in range(0, 10):
    print(data, " ", end="")
print()
for data in range(0, 10, 2):
    print(data, " ", end="")
print()

## For in list, dict, ...
list_data = [10, 20, 30, 40, 50]
for data in list_data:
    print(data, " ", end="")
print()
dict_data = {"key1":1, "key2":2}
for data in dict_data:
    print(data, " ", end="")
print()
for key, value in dict_data.items():
    print(key, " ", value)
print()

## list comprehension
list_data = [x**2 for x in range(5)]
print(list_data)

raw_data = [[1, 10], [2, 15], [3, 30], [4, 55]]

all_data = [x for x in raw_data]
x_data = [x[0] for x in raw_data]
y_data = [x[1] for x in raw_data]

print('all_data =', all_data)
print('x_data =', x_data)
print('y_data =', y_data)

### Quiz. Implement the code that outputs even numbers from 0 to 9 using list comprehension.
even_numbers = []
for data in range(10):
    if data % 2 == 0:
        even_numbers.append(data)
print(even_numbers)


## While
data = 5
while data >= 0:
    print("data ==", data)
    data -= 1

## break, continue
data = 5
while data >= 0:
    print("data ==", data)
    data -= 1
    if data == 2:
        print("break here.")
        break
    else:
        print("continue here.")
        continue
