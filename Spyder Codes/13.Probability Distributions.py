# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:48:36 2020

@author: Bhavin
"""
#Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math
#Writing probability mass functions for discrete variables

###############################################################################
#Bernoilli Distribution
# 1.1. Defining pmf
def my_bernoulli(x, p):
    """Bernoilli Distribution"""
    if x== 1:
        prob_x = p
    else:
        prob_x = 1-p
    return prob_x

# 2.Example of probability calculation
my_bernoulli(x=0, p = 0.3)
my_bernoulli(x=1, p = 0.3)

# 3.Example of pmf graph

x=np.array([0,1])

my_bernoulli_vect = np.vectorize(my_bernoulli)
prob_x = my_bernoulli_vect(x,0.6)

plt.scatter(x, prob_x)
plt.show()

# 4.Defining cumulative distribution function

def my_bernoulli_cdf(x,p):
    cdf_x = 0
    i=0
    while i <= x:
        cdf_x = cdf_x + my_bernoulli(i,p)
        i += 1
    return cdf_x

# 5.Example of cumulative probability calculation 
my_bernoulli_cdf(0,0.3)
my_bernoulli_cdf(1,0.3)

# 6.Example of cdf graph
x=np.array([0,1])

my_bernoulli_cdf_vect = np.vectorize(my_bernoulli_cdf)
cdf_x = my_bernoulli_cdf_vect(x,0.3)

plt.scatter(x, cdf_x)
plt.show()

plt.plot(x, cdf_x)
plt.show()

###############################################################################
#Discrete Uniform Distribution
# 1. Defining pmf
def my_discrete_uniform(x, a, b):
    """Discrete Uniform Distribution"""
    return 1/(b-a+1)

# 2.Example of probability calculation
my_discrete_uniform(4,1,6)

# 3.Example of pmf graph

n=6
x = np.arange(1,n+1)

my_discrete_uniform_vect = np.vectorize(my_discrete_uniform)

prob_x = my_discrete_uniform_vect(x,1,6)

plt.scatter(x, prob_x)
plt.show()

plt.plot(x, prob_x)
plt.show()

# 4.Defining cumulative distribution function

def my_discrete_uniform_cdf(x,a,b):
    cdf_x = 0
    i=a
    while i <= x:
        cdf_x = cdf_x + my_discrete_uniform(i,a,b)
        i += 1
    return cdf_x

# 5.Example of cumulative probability calculation 
my_discrete_uniform_cdf(4,1,6)
my_discrete_uniform_cdf(1,1,6)
my_discrete_uniform_cdf(6,1,6)
my_discrete_uniform_cdf(2,1,6)

# 6.Example of cdf graph
n=6
x=np.arange(1,n+1)

my_discrete_uniform_cdf_vect = np.vectorize(my_discrete_uniform_cdf)
cdf_x = my_discrete_uniform_cdf_vect(x,1,6)

plt.scatter(x, cdf_x)
plt.show()

plt.plot(x, cdf_x)
plt.show()

###############################################################################
#Binomial Distribution

# 1. Defining pmf

def factorial(n):
    """Python program to find the factorial of a number"""

    factorial = 1

    if n < 0:
        print("Sorry, factorial does not exist for negative numbers")
    elif n == 0:
        factorial = 1
    else:
        for i in range(1, n+1):
            factorial = factorial*i
    return factorial

factorial(0)
factorial(1)
factorial(2)
factorial(3)

def my_binomial(x,n,p):
    """Binomial Distribution"""
    return (factorial(n)/(factorial(n-x)*factorial(x)))*(p**x)*((1-p)**(n-x))

# 2.Example of probability calculation
my_binomial(6,10,0.5)

my_binomial(2,3,0.3)

my_binomial(0,3,0.3) + my_binomial(1,3,0.3) + my_binomial(2,3,0.3) + my_binomial(3,3,0.3)

# 3.Example of pmf graph

#Let n=100, p = 0.5, find out different probabilities of x
n=100
x = np.arange(0,n+1)

my_binomial_vect = np.vectorize(my_binomial) 
prob_x = my_binomial_vect(x, n=100, p=0.5)

plt.scatter(x, prob_x)
plt.show()

plt.plot(x, prob_x)
plt.show()

#Take n=10, p=0.5
n=10
x = np.arange(1,n+1)
prob_x = my_binomial_vect(x, n=10, p=0.5)

plt.plot(x, prob_x)
plt.show()

#Change the value of p for n=100
n=100
x = np.arange(1,n+1)

prob_x = my_binomial_vect(x, n=100, p=0.3)

plt.plot(x, prob_x)
plt.show()

#Multiple charts together
n=100
x = np.arange(1,n+1)

prob_x30 = my_binomial_vect(x, n=100, p=0.3)
prob_x70 = my_binomial_vect(x, n=100, p=0.7)
prob_x50 = my_binomial_vect(x, n=100, p=0.5)

plt.plot(x, prob_x30, color='g', label = 'p=0.3')
plt.plot(x, prob_x70, color='b', label = 'p=0.7')
plt.plot(x, prob_x50, color='r', label = 'p=0.5')
plt.legend()
plt.show()

# 4.Defining cumulative distribution function

def my_binomial_cdf(x,n,p):
    cdf_x = 0
    i=0
    while i <= x:
        cdf_x = cdf_x + my_binomial(i,n,p)
        i += 1
    return cdf_x

# 5.Example of cumulative probability calculation 
my_binomial_cdf(4,10,0.5)

# 6.Example of cdf graph
n=10
x=np.arange(0,n+1)

my_binomial_cdf_vect = np.vectorize(my_binomial_cdf)
cdf_x = my_binomial_cdf_vect(x,10,0.5)

plt.scatter(x, cdf_x)
plt.show()

plt.plot(x, cdf_x)
plt.show()

###############################################################################
"""
Quetion1: The probability that you will shoot a ballon with a gun is 0.3. 
Find out the cumulative probability of not being able to shoot a balloon.

Question2: Plot a graph of pmf of randomly drawing a card from a well sfuffeled 
deck of cards regardless of their pattern like spade, hearts etc.

Question 3: Suppose you have to cross three traffic signals on your way to work. 
The probability that you will see a red light at a signal is 0.3. What is the 
probability that you will see exactly 2 red lights on your way to work?

Question 4: From question 3 setup, what is the probability that you will not see 
all of them red together on your way to work?
"""
###############################################################################
#Poisson Distribution

# 1. Defining pmf
def my_poisson(x,m):
    """Poisson Distribution"""
    """m>0, x>=0"""
    return (m**x)*(math.exp(-m))/factorial(x)

# 2.Example of probability calculation
my_poisson(5,10)
my_poisson(5,100)
my_poisson(100,90)

"""
The Poisson distribution is popular for modeling the number of times an event 
occurs in an interval of time or space.

The Poisson distribution may be useful to model events such as

    The number of meteorites greater than 1 meter diameter that strike Earth in a year
    The number of patients arriving in an emergency room between 10 and 11 pm
    The number of laser photons hitting a detector in a particular time interval
"""

# 3.Example of pmf graph

#Let m=10, find out different probabilities of x
n=20
x = np.arange(0,n+1)

my_poisson_vect = np.vectorize(my_poisson) 
prob_x = my_poisson_vect(x, m=10)

plt.scatter(x, prob_x)
plt.show()

#Multiple charts together
n=20
x = np.arange(1,n+1)

prob_x1 = my_poisson_vect(x, m=1)
prob_x4 = my_poisson_vect(x, m=4)
prob_x10 = my_poisson_vect(x, m=10)

plt.plot(x, prob_x1, color='g', label = 'm=1')
plt.plot(x, prob_x4, color='b', label = 'm=4')
plt.plot(x, prob_x10, color='r', label = 'm=10')
plt.legend()
plt.show()

# 4.Defining cumulative distribution function

def my_poisson_cdf(x,m):
    cdf_x = 0
    i=0
    while i <= x:
        cdf_x = cdf_x + my_poisson(i,m)
        i += 1
    return cdf_x

# 5.Example of cumulative probability calculation 
my_poisson_cdf(4,5)

# 6.Example of cdf graph
n=10
x=np.arange(0,n+1)

my_poisson_cdf_vect = np.vectorize(my_poisson_cdf)
cdf_x = my_poisson_cdf_vect(x,5)

plt.scatter(x, cdf_x)
plt.show()

###############################################################################

#Continuous Uniform Distribution

# 1. Defining pdf
def my_conti_uniform(x,a,b):
    """Continuous Uniform Distribution"""
    """a<=x<=b, b!=a"""
    return 1/(b-a)

# 2.Example of probability calculation
my_conti_uniform(5,1,10)
my_conti_uniform(150,100,1000)

# 3.Example of pdf graph

#Let a=1, b=10, find out different probabilities of x

x = np.linspace(1,10,100)

my_conti_uniform_vect = np.vectorize(my_conti_uniform) 
prob_x = my_conti_uniform_vect(x, 1,10)

plt.scatter(x, prob_x)
plt.show()

#Multiple charts together
x = np.linspace(1,10,10)

prob_x_1_10 = my_conti_uniform_vect(x,1,10)
prob_x_0_15 = my_conti_uniform_vect(x, 0, 15)
prob_x_1_20 = my_conti_uniform_vect(x, 1, 20)

plt.plot(x, prob_x_1_10, color='g', label = 'a=1, b=10')
plt.plot(x, prob_x_0_15, color='b', label = 'a=0, b=15')
plt.plot(x, prob_x_1_20, color='r', label = 'a=1,b=20')
plt.legend()
plt.show()

# 4.Defining cumulative distribution function

import sympy as sp
x = sp.Symbol('x')
a = sp.Symbol('a')
b = sp.Symbol('b')
     
sp.integrate(my_conti_uniform(x,a,b),(x,a,x))

def my_conti_uniform_cdf(x,a,b):
    return (x-a)/(b-a)

# 5.Example of cumulative probability calculation 
my_conti_uniform_cdf(4,1,5)

# 6.Example of cdf graph
x=np.linspace(1,5,10)

my_conti_uniform_cdf_vect = np.vectorize(my_conti_uniform_cdf)
cdf_x = my_conti_uniform_cdf_vect(x,1,5)

plt.scatter(x, cdf_x)
plt.show()

###############################################################################
#Continuous Normal Distribution

# 1. Defining pdf
def my_normal(x,m,s):
    """Normal Distribution"""
    return math.exp((-1/2)*((x-m)/s)**2/(s*math.sqrt(2*math.pi)))

# 2.Example of probability calculation
my_normal(30,35,20)
my_normal(30.5,35,20)
my_normal(0.5,0,1)

# 3.Example of pdf graph

#Let m=35, s=20, find out different probabilities of x

x = np.linspace(0,100,500)

my_normal_vect = np.vectorize(my_normal) 
prob_x = my_normal_vect(x, 35,20)

plt.scatter(x, prob_x)
plt.show()

#Let m=0, s=1, find out different probabilities of x

x = np.linspace(-1,1,50)
prob_x = my_normal_vect(x,0,1)

plt.scatter(x, prob_x)
plt.show()

#Multiple charts together
x = np.linspace(-5,5,1000)

#Find probability curves for pairs of (m,s**2)
(0,0.2)
(0,1)
(0,5)
(-2,0.5)

prob_x_0_point2 = my_normal_vect(x,0,math.sqrt(0.2))
prob_x_0_1 = my_normal_vect(x,0,1)
prob_x_0_5 = my_normal_vect(x,0,math.sqrt(5))
prob_x_neg2_point5 = my_normal_vect(x,-2,math.sqrt(0.5))

plt.plot(x, prob_x_0_point2, color='g', label = '(m,s**2)=(0,0.2)')
plt.plot(x, prob_x_0_1, color='b', label = '(m,s**2)=(0,1)')
plt.plot(x, prob_x_0_5, color='r', label = '(m,s**2)=(0,5)')
plt.plot(x, prob_x_neg2_point5, color='y', label = '(m,s**2)=(-2,0.5)')

plt.legend()
plt.show()

# 4.Defining cumulative distribution function

#Really out of the scope for this course
     
from scipy.stats import norm

def my_normal_cdf(x):
    return norm.cdf(x) #Standard normal distribution

# 5.Example of cumulative probability calculation 
my_normal_cdf(0)

# 6.Example of cdf graph
x=np.linspace(1,5,25)

my_normal_cdf_vect = np.vectorize(my_normal_cdf)
m = np.mean(x)
s = np.std(x)

std_x = (x - m)/s

std_m = np.mean(std_x)
std_s = np.std(std_x)

cdf_x = my_normal_cdf_vect(std_x)

plt.scatter(std_x, cdf_x)
plt.show()

###############################################################################

#Find out the following probabilities

"""
Question 1: The average number of homes sold by the Acme Realty company is 
2 homes per day. What is the probability that exactly 3 homes will be sold tomorrow? 

Question 2: Suppose the average number of lions seen on a 1-day safari is 5. 
What is the probability that tourists will see fewer than four lions on 
the next 1-day safari? 

Question 3: An average light bulb manufactured by the Acme Corporation lasts 
300 days with a standard deviation of 50 days. Assuming that bulb life is 
normally distributed, what is the probability that an Acme light bulb will 
last at most 365 days?

Question 4: Suppose scores on an IQ test are normally distributed. If the test 
has a mean of 100 and a standard deviation of 10, what is the probability that 
a person who takes the test will score between 90 and 110?

Question 5: Molly earned a score of 940 on a national achievement test. 
The mean test score was 850 with a standard deviation of 100. 
What proportion of students had a higher score than Molly? 
(Assume that test scores are normally distributed.)

Question 6: What is the probability of obtaining 45 or fewer heads in 100 
tosses of a coin?

Question 7: The probability that a student is accepted to a prestigious college 
is 0.3. If 5 students from the same school apply, what is the probability that 
at most 2 are accepted? 

Question 8: You have been given that X follows uniform distribution with 
(a=100,b=300). Calculate P(x>174) and P(100<x<226)

Question 9: A random variable X is uniformly distributed between 32 and 42. 
What is the probability that X will be between 32 and 40?

Question 10: Find the mean and standard deviation for questions 1, 2, 3, 6, 8 and 9.
Intrepret the results.

Question 11: Ace Heating and Air Conditioning Service finds that the amount of 
time a repairman needs to fix a furnace is uniformly distributed between 1.5 
and four hours. Let x = the time needed to fix a furnace. Then x ~ U (1.5, 4).
Find the probability that a randomly selected furnace repair requires more 
than two hours. Find the probability that a randomly selected furnace repair 
requires less than three hours. Find the mean and standard deviation
"""