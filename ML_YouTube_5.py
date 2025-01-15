# class, exception, with
## class => class class_name:
##            def __init__(self, input1, input2, ...):      # Constructor(생성자)
##            def method_name(self, input1, input2, ...):   # Method(메소드)
class Person:
    def __init__(self, name):
        self.name = name
        print(self.name + " is initialized.")

    def work(self, company):
        print(self.name + " is working in " + company)

    def sleep(self):
        print(self.name + " is sleeping")

obj = Person("PARK")

obj.work("ABCDEF")

obj.sleep()

print("current person object count is ", obj.name)\

## class variable, class method
class Person:
    count = 0
    def __init__(self, name):
        self.name = name
        Person.count += 1
        print(self.name + " is initialized.")

    def work(self, company):
        print(self.name + " is working in " + company)

    def sleep(self):
        print(self.name + " is sleeping")
    
    @classmethod
    def getCount(cls):
        return cls.count

obj1 = Person("PARK")
obj2 = Person("KIM")

obj1.work("ABCDEF")
obj2.sleep()

print("current person object count is ", obj1.name, ", ", obj2.name)
print("Person count ==", Person.getCount())
print(Person.count)

## Private variable
class PrivateMemberTest:
    def __init__(self, name1, name2):
        self.name1 = name1
        self.__name2 = name2
        print("initialized with " + name1 + ", " + name2)

    def getNames(self):
        self.__printNames()
        return self.name1, self.__name2

    def __printNames(self):
        print(self.name1, self.__name2)

obj = PrivateMemberTest("PARK", "KIM")

print(obj.name1)
print(obj.getNames(), '\n')
#print(obj.__printNames())  # error
#print(obj.__name2)         # error

#@ ex.
def print_name(name):
    print("[def] ", name)
class SameTest:
    def __init__(self):
        pass     # pass is a null operation. It does nothing.

    # Define a method with the same name as an external function.
    def print_name(self, name):
        print("[SameTest] ", name)
    def call_test(self):
        print_name("KIM")       # Call an external function.
        self.print_name("KIM")  # Call a method in the class.
obj = SameTest()
print_name("LEE")
obj.print_name("LEE")
obj.call_test()


## Exception
## try ~ except ~ finally => try:
##                               # Code that may cause an error.
##                           except Excpetion type:
##                               # Code that handles when exceptions occur.
##                           else:
##                               # Code that is executed when no exceptions occur. (optional)
##                           finally:
##                               # Code that is executed regardless of whether 
##                                 an exception occurs. (optional)
def calc(list_data):
    sum = 0
    try:            # It is executed when an exception occurs.
        sum = list_data[0] + list_data[1] + list_data[2]
        if sum < 0:
            raise Exception("Sum is minus.")    # Raise an exception. Debugging is easier.
    except IndexError as err:
        print(str(err))
    except Exception as err:
        print(str(err))
    finally:        # It is executed regardless of whether an exception occurs.
        print(sum)

calc([1, 2])
calc([1, 2, -100])


## with: [open() => read() or write() => close()] => automatically close the file.
#@ ex. General file open, read, close
f = open("./file_test", 'w')
f.write("Hello Python!!!")
f.close()

#@ ex. with open
with open("./file_test", 'w') as f:
    f.write("Hello Python!!!")
