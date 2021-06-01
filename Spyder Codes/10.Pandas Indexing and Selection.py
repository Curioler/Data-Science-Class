# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:04:06 2020

@author: Bhavin
"""

import seaborn as sns
import pandas as pd
import numpy as np

"""
Pandas Indexing and Selecting

Let's talk about slicing and dicing pandas data. We are going to be going over four topics:

    Review the basics
    Multi-index
    Getting Single Values
    Pointing out some stuff you don't need to worry about

As always you can check out the full documentation: basic indexing and 
advanced indexing. But be warned that they are very long and tell you way more 
than you'd need to know :)
    
https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html

https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
"""

###############################################################################

#Review the Basics

"""
Series

Pandas has two data structures a Series and a Dataframe. A series is like a 
column in excel, basically a list of datapoints all of the same type. 
And the basic way to create a series object is below:
"""

pd.Series?

s = pd.Series(
        np.random.randn(5), 
        index=['a', 'b', 'c', 'd', 'e'], 
        name='example')

s

"""
There are other ways to make a series (like from a dictionary), but in general 
this is the one that is used. So notice that a series has basically three 
important parts:

    The data
    The index
    The name

The data can be a list of data, or a single instance that broadcasts, like below:
"""

pd.Series(5, index=['a', 'b', 'c', 'd', 'e'])

"""
The data is basically what you as a data scientist are interested in. 
The index is often used in time series, but otherwise the index is not really 
used for series (index is really used for dataframes quite a lot!). 
But notice that each datapoint is associated with an index.

Finally the name. 
The name is only really important when you add a series to a dataframe. 
In that case the name of the series becomes the column.

You have so far not seen why series are all that useful, but now we start to 
get into it. Series have various ways that you can index into them:
"""

s[0]

s[:3]

s[[4, 3, 1]]

s.values

type(s.values)

s['e'] = 500
s

#Frequently used
s[[True, True, False, False, True]]

# or the extremely common
s[s > 0], s > 0

# and you can mutate the data too
# you'll just need to be careful with this!
s[s < 0] *= -1
s

"""
one thing that is super useful about series is that you can do vectorized 
operations (fast computations on everything in the entire series) on them. 
And you have already seen one.
"""

s > 0

s + s

np.exp(s)

s.mean()

# just be careful with some operations
# if the indexes don't match up you will get nans
s + s[s > 1]

"""
These types of operations that are over columns is what pandas is made for. 
Any time you stray from doing operations over columns, you should think to 
yourself: is pandas the best tool for me?
"""
###############################################################################
"""
Now doing operations over one column might seem useful, 
but what about operations over multiple columns.

DataFrames

Series are nice, but the really nice thing about them is dataframes. 
Dataframes are like an entire excel spreadsheet! 
As you can probably guess, dataframes are a list of series, 
each one with a name and the same index. 
Thus an easy way to create a dataframe is to create it with a 
dictionary of series/lists:
"""

d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)

df

#Using numpy arrays

d = {'one' : 'Hellow',
    'two' : np.array([1., 2., 3., 4.])}

df = pd.DataFrame(d)
df

#Updating column names and index
df.columns = ['Col_1', 'Col_2']
df.index = ['a', 'b', 'c', 'd']
df

#Just like the series, you can do all the operations
#Get a column

df['Col_1']

#delete a column
del df['Col_1']
df

#Broadcasting columns

df['Col_3'] = df['Col_2'] + df['Col_2']
df['Col_4'] = 'four'
df['Col_5'] = df['Col_4'][:2]
df

#Get more than one columns

df[['Col_2','Col_3']]

# select by indexes and column names
df.loc['a', 'Col_2']

df.loc['d':'a':-1, 'Col_2':'Col_3']

# select rows and columns by their ordering
df.iloc[1:3, 0]

df.iloc[1:3]

"""
DataFrame Functions

In addition to doing these columnwise operations, you can also do some dataframewise operations.

The most useful of these is the copy method, it makes a copy :)
"""
df.copy()

#The astype method converts the data types of columns

df.Col_2.astype(np.int)

#Transpose a dataframe

df.T

"""
This puts the rows as the columns and the columns as the rows. It can be a 
good way to do row-wise operations, but mainly I do it to display dataframe 
values. Below are the three common ways to display dataframe values:
"""

df.head(2)

df.info()

df.describe(include='all')

df.describe()

#Messy display
for i in range(20):
    df[i] = i
    
df.head()

#Transposing helps


#transposing helps
df.head().T

df.head().T.head()

"""
Sometimes this will also truncate. To view more you can always change the view 
options as below (btw, there are many many options in pandas, you can check 
them all out either here (https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html) 
or with a pd.set_option?):
"""
pd.set_option('display.max_rows', 100)
pd.set_option('precision', 7)

###############################################################################

#Let's get the good old tips data

tips = sns.load_dataset('tips')
tips.head(3)

# 1) get columns
tips[['total_bill', 'tip']].head()

# 2) get some rows
tips[3:5]

# 3) select rows and columns based on their name
tips.loc[2:4, 'sex': 'day']

# select rows and columns by their ordering
tips.iloc[1:3, 0:2]

# 5) select using a bool series
tips[tips['tip'] > 1].head()

"""
But this is just the tip of the iceberg (well actually it's 90% of the iceberg).

But there are a couple of other important concepts that you will most likely 
get into when diving into other pandas functionalities.
"""

"""
Multi-index

A subject that you might not think that you'd need - but turns out to be 
a rather frequent usecase.

The initial idea behind the multi-index was to provide a framework to work 
with higher dim data (and thus a replacement for panels).

But because of some operations it became quite common place. 
In almost all cases multi-index comes from groupby's 
(you will almost never construct it or read it in yourself).

Let's do an example below:
"""
mi_tips = tips.groupby(['sex', 'smoker']).agg({'tip': 'mean'})
mi_tips

mi_tips.index

"""
Ultimately there are a ton of operations that you can do on top of this type 
of data. And there are equivalent multi-index operations you can do, like this:
"""

mi_tips.loc[('Male', 'No')]

mi_tips = tips.groupby(['sex', 'smoker']).agg({'total_bill':'mean','tip': 'mean'})
mi_tips

mi_tips.loc[('Male', 'No')]

#Simplest way is to reset the index
ri_tips = mi_tips.reset_index()
ri_tips

"""
Notice how we get values spread out over the full column now. So in this way 
it is easy to select only the male non-smokers:
"""
ri_tips[(ri_tips['smoker'] == 'No') & (ri_tips['sex'] == 'Male')]

#Another way you can deal with this is to only certain indexes out:

ri0_tips = mi_tips.reset_index(level=0)
ri0_tips.loc['Yes']

ri1_tips = mi_tips.reset_index(level=1)
ri1_tips.loc['Male']

"""
And finally you can pull indexes back into the index (basically only useful 
for certain types of merges).
"""

ri_tips.set_index(['sex', 'smoker'])

ri0_tips
ri0_tips.set_index('sex', append=True)

###############################################################################

"""
Getting Single Values

The next little indexing trick is one that is mostly about speed. But it is 
getting and setting single values. It is pretty simple:
"""

#When getting/setting single values you should use the at function

tips.at[0, 'total_bill'] = 9000
tips.head(3)

tips.iat[0, 0]

"""
Where, Masks and Queries

These are things that are built into pandas. They are pretty redundant and 
don't happen too often.

They are a bit faster, yes. But the mental space is probably not worth it. 
So if you wanna learn it, go for it. If not, probably won't matter.

Let me show you how you'd duplicate mask functionality below.
"""

df = pd.DataFrame(np.random.randn(25).reshape((5, 5)))
df.head()

df.where(df > 0)

df[df < 0] = np.NaN
df

df[df < 0] = 9999
df

