# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:38:53 2020

@author: Bhavin
"""

"""
vectorize – Make a scalar function work on vectors

With the help of vectorize() you can make a function that is meant to work on individual numbers, 
to work on arrays."""

# Define a scalar function
def foo(x):
    if x % 2 == 1:
        return x**2
    else:
        return x/2

# On a scalar
print('x = 10 returns ', foo(10))
print('x = 11 returns ', foo(11))

# On a vector, doesn't work
print('x = [10, 11, 12] returns ', foo([10, 11, 12]))  # Error 

# Vectorize foo(). Make it work on vectors.
import numpy as np
foo_v = np.vectorize(foo, otypes=[float])

print('x = [10, 11, 12] returns ', foo_v([10, 11, 12]))
print('x = [[10, 11, 12], [1, 2, 3]] returns ', foo_v([[10, 11, 12], [1, 2, 3]]))

###############################################################################


"""apply_along_axis – Apply a function column wise or row wise

It takes as arguments:

    Function that works on a 1D vector (func1d)
    Axis along which to apply func1d. For a 2D array, 1 is row wise and 0 is column wise.
    Array on which func1d should be applied.

"""
# Create a 4x10 random array
np.random.seed(100)
arr_x = np.random.randint(1,10,size=[4,10])
print(arr_x)

# Define func1d (function to find the range)
def max_minus_min(x):
    return np.max(x) - np.min(x)

# Apply along the rows
print('Row wise: ', np.apply_along_axis(max_minus_min, 1, arr=arr_x))

# Apply along the columns
print('Column wise: ', np.apply_along_axis(max_minus_min, 0, arr=arr_x))

###############################################################################
"""
searchsorted – Find the location to insert so the array will remain sorted

It gives the index position at which a number should be inserted in order to keep the array sorted.
"""
# example of searchsorted
x = np.arange(10)
print(x)
print('Where should 5 be inserted?: ', np.searchsorted(x, 5))
print('Where should 5 be inserted (right)?: ', np.searchsorted(x, 5, side='right'))

###############################################################################
"""
Sometimes you might want to convert a 1D array into a 2D array (like a spreadsheet) 
without adding any additional data. 

You can do this by inserting a new axis using the np.newaxis.
"""

# Create a 1D array
x = np.arange(5)
print('Original array: ', x)

# Introduce a new column axis
x_col = x[:, np.newaxis]
print('x_col shape: ', x_col.shape)
print(x_col)

# Introduce a new row axis
x_row = x[np.newaxis, :]
print('x_row shape: ', x_row.shape)
print(x_row)

###############################################################################
"""
Digitize

Use np.digitize to return the index position of the bin each element belongs to.
"""

# Create the array and bins
x = np.arange(10)
bins = np.array([0, 3, 6, 9])

# Get bin allotments
np.digitize(x, bins)

###############################################################################
"""
Clip

Use np.clip to cap the numbers within a given cutoff range.
All number lesser than the lower limit will be replaced by the lower limit.
Same applies to the upper limit also.
"""
# Cap all elements of x to lie between 3 and 8
np.clip(x, 3, 8)

###############################################################################

"""
What is missing in numpy?

So far we have covered a good number of techniques to do data manipulations with numpy.
But there are a considerable number of things you can’t do with numpy directly.
At least to my limited knowledge. Let me list a few:

    No direct function to merge two 2D arrays based on a common column.
    Create pivot tables directly
    No direct way of doing 2D cross tabulations.
    No direct method to compute statistics (like mean) grouped by unique values in an array.
    And more..

Well, the reason I am telling you this is these shortcomings are nicely handled 
by the spectacular pandas library."""