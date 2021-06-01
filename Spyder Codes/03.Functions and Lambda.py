# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:15:52 2020

@author: Bhavin
"""
#Example of a function without any parameter
def my_function():
  print("Hello from a function") 

#Calling a function
def my_function():
  print("Hello from a function")

my_function()

#Function parameters/arguments

def my_function(fname):
  print(fname + " Refsnes")

my_function("Emil")
my_function("Tobias")
my_function("Linus") 

#Multiple arguments

def my_function(fname, lname):
  print(fname + " " + lname)

my_function("Emil", "Refsnes") 

def my_function(fname, lname):
  print(fname + " " + lname)

my_function("Emil") #check what happens

"""
Arbitrary Arguments, *args

If you do not know how many arguments that will be passed into your function, 
add a * before the parameter name in the function definition.

This way the function will receive a tuple of arguments, and can access the items accordingly:
    
Arbitrary Arguments are often shortened to *args in Python documentations.
"""

def my_function(*kids):
  print("The youngest child is " + kids[2])

my_function("Emil", "Tobias", "Linus") 


"""
You can also send arguments with the key = value syntax.

This way the order of the arguments does not matter.
"""

def my_function(child3, child2, child1):
  print("The youngest child is " + child3)

my_function(child1 = "Emil", child2 = "Tobias", child3 = "Linus") 

"""
Arbitrary Keyword Arguments, **kwargs

If you do not know how many keyword arguments that will be passed into your function,
 add two asterix: ** before the parameter name in the function definition.

This way the function will receive a dictionary of arguments, and can access the items accordingly:

Arbitrary Kword Arguments are often shortened to **kwargs in Python documentations.
"""
def my_function(**kid):
  print("His last name is " + kid["lname"])

my_function(fname = "Tobias", lname = "Refsnes") 

#Default Parameter value

def my_function(country = "Norway"):
  print("I am from " + country)

my_function("Sweden")
my_function("India")
my_function()
my_function("Brazil") 

#Passing a List as an Argument

def my_function(food):
  for x in food:
    print(x)

fruits = ["apple", "banana", "cherry"]

my_function(fruits)

#Return values

def my_function(x):
  return 5 * x

print(my_function(3))
print(my_function(5))
print(my_function(9)) 

"""
function definitions cannot be empty, 
but if you for some reason have a function definition with no content, 
put in the pass statement to avoid getting an error.
"""

def myfunction():
  pass

"""
Recursion

Python also accepts function recursion, which means a defined function can call itself.

Recursion is a common mathematical and programming concept.
It means that a function calls itself.
This has the benefit of meaning that you can loop through data to reach a result.

The developer should be very careful with recursion as it can be quite easy 
to slip into writing a function which never terminates, 
or one that uses excess amounts of memory or processor power. 
However, when written correctly recursion can be a very efficient 
and mathematically-elegant approach to programming.

In this example, tri_recursion() is a function that we have defined to call itself ("recurse").
We use the k variable as the data, which decrements (-1) every time we recurse.
The recursion ends when the condition is not greater than 0 (i.e. when it is 0).

To a new developer it can take some time to work out how exactly this works, 
best way to find out is by testing and modifying it.
"""

def tri_recursion(k):
  if(k>0):
    result = k+tri_recursion(k-1)
    print(result)
  else:
    result = 0
  return result

print("\n\nRecursion Example Results")
tri_recursion(6)

###############################################################################

#Example of recursion function


# An example of a recursive function to
# find the factorial of a number

def calc_factorial(x):
    """This is a recursive function
    to find the factorial of an integer"""

    if x == 1:
        return 1
    else:
        return (x * calc_factorial(x-1))

num = 4
print("The factorial of", num, "is", calc_factorial(num))

""""
How it works?
    calc_factorial(4)              # 1st call with 4
    4 * calc_factorial(3)          # 2nd call with 3
    4 * 3 * calc_factorial(2)      # 3rd call with 2
    4 * 3 * 2 * calc_factorial(1)  # 4th call with 1
    4 * 3 * 2 * 1                  # return from 4th call as number=1
    4 * 3 * 2                      # return from 3rd call
    4 * 6                          # return from 2nd call
    24                             # return from 1st call
"""


"""
Write the following recursion functions:
    Sum of fibonnaci sequence
"""

###############################################################################
"""
A lambda function is a small anonymous function.

A lambda function can take any number of arguments, but can only have one expression.

syntax:
lambda arguments : expression
"""

#one argument
x = lambda a : a + 10
print(x(5)) 

#two arguments
x = lambda a, b : a * b
print(x(5, 6))

#three arguments

x = lambda a, b, c : a + b + c
print(x(5, 6, 2)) 

#use of lambda function inside a function definition
def myfunc(n):
  return lambda a : a * n 

#Exercises
#Datasets https://vincentarelbundock.github.io/Rdatasets/datasets.html
  
"""
Use the acme dataset to answer the following questions:
    
Define a function to calculate the total profit between two periods
Using the function to 
 - calculate the profit for Acme Cleveland Corporation at the end of 1988
 - calculate the profit for the market at the end of 1988
 - how did ACC do compared to the market at the end of 1990?
 - write a function to plot a line chart
 - Use the charting function to plot price chart for ACC and Market
 - write a lambda function to find the share price of ACC at the end of a month
 - use the lambda function to find the share price at the end of March for every 
     year considering that the price was 100 USD at the end of December 1985."""

file = r"https://vincentarelbundock.github.io/Rdatasets/csv/boot/acme.csv"

import numpy as np

data = np.genfromtxt(file,skip_header=1,delimiter=',',usecols = (2, 3) )

market = data[:,0]
acme = data[:,1]

def profit(cmp=100, instrument=acme, p1=37, p2=61):
    slice = instrument[p1:p2]
    initial_price = cmp
    for change in slice:
        month_profit = cmp*change
        cmp += month_profit
        profit = cmp - initial_price
    return profit
x = profit(instrument=acme)
print(x)


def profit(cmp=100, instrument=acme, p1=37, p2=61):
    slice = instrument[p1:p2]
    print(slice)
    change_array = slice*cmp
    profit = np.sum(change_array)
    return profit
x = profit(instrument=acme)
print(x)