# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:33:03 2020

@author: Bhavin
"""

import seaborn as sns
import pandas as pd
import numpy as np

"""
Pandas Group Operations

Let's next go over grouped operations with pandas. This section of the pandas 
library does not have as much feature bloat as other parts, which is nice. 
And the community is starting to narrow around a couple of operations that are 
core to grouped operations. We'll be going over these operations with particular 
emphasis on groupby and agg:

    groupby
    agg
    filter
    transform

Check out the full documentation here, but be warned it is a bit long :)
http://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
        
Let's start with our good old tips dataset:
"""

tips = sns.load_dataset('tips')
tips.head(3)

"""
Groupby

A grouped operation starts by specifying which groups of data that we would 
want to operate over. There are many ways of making groups, but the tool that 
pandas uses to make groups of data, is groupby
"""
tips_gb = tips.groupby(['sex', 'smoker'])
tips_gb

"""
Groupby works by telling pandas a couple of columns. Pandas will look in your 
data and see every unique combination of the columns that you specify. 
Each unique combination is a group. So in this case we will have 
four groups: male smoker, female smoker, male non-smoker, female non-smoker.

The groupby object by itself is not super important.

Once we have these groups (specified in the groupby object), we can do three 
types of operations on it (with the most important being agg)

Agg:
The aggregate operation aggregates all the data in these groups into one value. 
You use a dictionary to specify which values you'd like. For example look below, 
we are asking for both the mean and the min value of the tip column for each group:
"""

tips_agg = tips_gb.agg({'tip': 'mean','day': 'first','total_bill': 'size'})
tips_agg

#Is it too difficult? Let's break it down

tips.groupby('sex')['total_bill'].sum()
tips.groupby('sex')['tip'].sum()

tips.groupby('smoker')['total_bill'].sum()
tips.groupby('smoker')['tip'].sum()

tips.groupby('sex')['total_bill'].mean()
tips.groupby('sex')['tip'].mean()

tips.groupby('smoker')['total_bill'].mean()
tips.groupby('smoker')['tip'].mean()

#Find the total bill and tip amount by day, time and size

#Let's go by multiple columns

tips.groupby(['sex','smoker'])['total_bill'].sum()

tips.groupby(['sex','smoker'])['total_bill'].max()

tips.groupby('day').count()

tips.groupby('day')['total_bill'].count()

#Find the count of bills by day and sex
#Find the count of bills by time
#Find the count of bills by size
#Find the count of bills by day and time

tips.groupby(['sex','smoker'])['total_bill','tip'].sum()

tips.groupby(['sex','smoker'])['total_bill','tip'].mean()

tips.groupby(['sex','smoker'])['total_bill','tip'].min()

#Find the minimum amount of tip by day and time
#Find the minimum amount of total bill by day and time

#Find the maximum amount of tip by day and time
#Find the maximum amount of total bill by day and time

#Some important attributes
tips.groupby(['sex', 'smoker']).groups

tips.groupby('sex')['total_bill'].first()
tips.groupby('sex')['total_bill'].last()

tips.groupby('smoker')['total_bill'].first()
tips.groupby('smoker')['total_bill'].last()

#Find the first and last bill by day

"""
Note:
- By default, the axis = 0, you can use axis = 1 too
- By default, sort='True', you can use sort = 'False' too 
"""

tips.groupby('sex').get_group('Male')
tips.groupby('smoker').get_group('Yes')

tips.groupby(['sex','smoker']).get_group(('Male','Yes'))

len(tips.groupby(['sex','smoker']))

#Don't stop here, just go crazy and be creative

import matplotlib.pyplot as plt
tips.groupby('sex').boxplot()
plt.show()

tips.groupby('sex')['total_bill'].describe()
tips.groupby('sex')['tip'].describe()

#Draw the boxplot by smoker for tip
#Draw the boxplot by size for tip
#Describe the total bill by day
#Describe the tip by day

"""
You must try these:
count, dtype, hist, median, size, std, var
"""

#Iterating through groups

tips_grouped = tips.groupby('sex')

for name, group in tips_grouped:
    print(name)
    print(group['total_bill'].sum())
    
#In the case of grouping by multiple keys, the group name will be a tuple:
    
tips_grouped = tips.groupby(['sex','smoker'])

for name, group in tips_grouped:
    print(name)
    print(group['total_bill'].sum())

#Aggregation
    
"""
Once the GroupBy object has been created, several methods are available to perform 
a computation on the grouped data. These operations are similar to the aggregating 
API, window functions API, and resample API.

An obvious one is aggregation via the aggregate() or equivalently agg() method:
"""

tips.groupby('sex').aggregate(np.sum)
tips.groupby('sex')['total_bill'].aggregate(np.sum)

tips.groupby('sex').agg(np.sum)
tips.groupby('sex')['total_bill'].agg(np.sum)

def cv(x):
    return np.std(x)/np.mean(x)

tips.groupby('sex').agg(cv)

#Can we use a lambda function?
tips.groupby('sex').agg(lambda x: np.std(x)/np.mean(x))

#Define a function to calculate coefficient of range
#Find the coeefficient of range for total_bill, tip and size by sex and smoker

tips.groupby('sex', as_index=False).agg(cv)
tips.groupby(['sex','smoker'], as_index=False).agg(cv)

"""
Note that you could use the reset_index DataFrame function to achieve the same 
result as the column names are stored in the resulting MultiIndex:
"""

"""
Commonly used agg functions
mean(): Compute mean of groups
sum(): Compute sum of group values
size(): Compute group sizes
count(): Compute count of group
std(): Standard deviation of groups
var(): Compute variance of groups
sem(): Standard error of the mean of groups
describe(): Generates descriptive statistics
first(): Compute first of group values
last(): Compute last of group values
nth(): Take nth value, or a subset if n is a list
min(): Compute min of group values
max(): Compute max of group values

The aggregating functions above will exclude NA values. Any function which 
reduces a Series to a scalar value is an aggregation function and will work
"""

"""
With grouped Series/DataFrame you can also pass a list or dict of functions to do 
aggregation with, outputting a DataFrame
"""

tips.groupby('sex', as_index=False)['size'].agg([np.mean, np.std, cv])
tips.groupby(['sex','smoker'], as_index=False)['tip'].agg([np.mean, np.std, cv])
tips.groupby(['sex','smoker'], as_index=False)['total_bill'].agg([np.mean, np.std, cv])

#If you need different names
tips.groupby(['sex','smoker'], as_index=False)['total_bill'].agg(
        [np.mean, np.std, cv]).rename(
                 columns={'mean':'Average',
                 'std':'Standard Deviation',
                 'cv':'Coefficient of variation'}
                )

#Try to write the full code in one line and then break it on to several lines

#Just using the lambda functions
tips.groupby('sex')['size'].agg(
        [lambda x: x.max() - x.min(),
        lambda x: x.median() - x.mean(),
        lambda x: np.std(x)/np.mean(x)])

#Create two lambda functions 1) mean - std and 2) mean + std

#naming the columns
tips.groupby('sex').agg(
        size_max = pd.NamedAgg(column = 'size', aggfunc = 'max'),
        size_min = pd.NamedAgg(column='size', aggfunc = 'min'),
        size_avg = pd.NamedAgg(column = 'size', aggfunc = 'mean'))

tips.groupby('sex').agg(
        size_max = pd.NamedAgg(column = 'size', aggfunc = 'max'),
        size_min = pd.NamedAgg(column='size', aggfunc = 'min'),
        size_avg = pd.NamedAgg(column = 'size', aggfunc = np.mean),
        size_std = pd.NamedAgg(column = 'size', aggfunc = np.std),
        size_cv = pd.NamedAgg(column = 'size', aggfunc = cv)
        )

#pandas.NamedAgg is just a namedtuple. Plain tuples are allowed as well.

tips.groupby('smoker').agg(
        size_max = pd.NamedAgg('size', 'max'),
        size_min = pd.NamedAgg('size', 'min'),
        size_avg = pd.NamedAgg('size', np.mean),
        size_std = pd.NamedAgg('size', np.std),
        size_cv = pd.NamedAgg('size', cv)
        )
#working with series

tips.groupby('smoker').tip.agg(
        tip_max = 'max',
        tip_min = 'min',
        tip_avg = np.mean,
        tip_std = np.std,
        tip_cv = cv
        )

tips.groupby('smoker').agg({
        'total_bill' : 'max',
        'tip' : 'max',
        'size' : 'count'
        })

tips.groupby(['sex', 'smoker']).agg({'tip': 'mean','day': 'first','total_bill': 'size'})

###############################################################################

"""
Filter

The next common group operation is a filter. This one is pretty simple, we 
filter out member of groups that don't meet our criteria.

For example let's only look at the least busy times the place is open. 
One way we might do that is exclude all times above the median from the analysis
"""    

# we use the exact same groupby syntax
tips_gb = tips.groupby(['day', 'time'])

median_size = tips_gb.agg({'size': 'sum'}).median()[0]
median_size    

tips_gb.agg({'size': 'sum'})
tips_gb.agg({'size': 'sum'}).median()

tips_gb.agg({'size': 'sum', 'total_bill':'sum'})
tips_gb.agg({'size': 'sum', 'total_bill':'sum'}).median()
tips_gb.agg({'size': 'sum', 'total_bill':'sum'}).median()[0]

# notice that we carved out quite a few rows
tips_gb.filter(lambda group: group['size'].sum() < median_size).head()

tips_gb.filter(lambda group: group['size'].sum() < median_size)

#The filter method returns a subset of the original object.
tips.groupby(['day', 'time']).filter(lambda x: x.tip.mean() > 3)
tips.groupby(['sex', 'smoker']).filter(lambda x: x.tip.mean() > 3)

tips.groupby(['sex', 'smoker']).filter(lambda group: group.tip.mean() > 3)
tips.groupby(['sex', 'smoker']).filter(lambda group: group.tip.mean() > 3).shape

tips.groupby(['sex', 'smoker']).tip.mean()

tips.groupby(['sex', 'smoker']).agg({'tip': 'mean', 'size' : 'count'}).groupby('sex').sum()
#break this statement if you want

"""
Note:
The argument of filter must be a function that, applied to the group as a 
whole, returns True or False.
"""

tips.groupby(['sex', 'smoker']).filter(lambda group: group.tip.mean() > 3, dropna=False)

###############################################################################

"""
Transform
The final group operation is transform. This uses group information to apply 
transformations to individual data points. For example look below: each day 
let's divide by the bill and tip by the average amount spent on that day. 
That way we can look at how much that bill differs from the average of that day
"""

tips_gb = tips.groupby(['day'])

tips_gb[['total_bill', 'tip']].transform(lambda x: x / x.mean()).head()


tips.groupby(['day']).transform(lambda group: group.max() - group.min())

###############################################################################

#Exercises

dtypes = {
    "first_name": "category",
    "gender": "category",
    "type": "category",
    "state": "category",
    "party": "category",
}

df = pd.read_csv(
    r"C:\Users\Bhavin\Google Drive\Data Science Class\Assets\Groupby dataset\legislators-historical.csv",
    dtype=dtypes,
    usecols=list(dtypes) + ["birthday", "last_name"],
    parse_dates=["birthday"]
)

"""
The dataset contains members’ first and last names, birth date, gender, type 
("rep" for House of Representatives or "sen" for Senate), U.S. state, and political party.
1. Explore the dataframe /You can see that most columns of the dataset have 
   the type category, which reduces the memory load on your machine.
2. Get the number of politicians by state
3. Get the number of politicians by gender
4. Get all the entries for the state 'PA'
5. Get the number of politicians by type
6. Get the number of politicians by type and party
"""