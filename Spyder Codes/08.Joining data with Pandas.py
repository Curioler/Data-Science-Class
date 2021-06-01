# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 07:28:04 2020

@author: Bhavin
"""

import pandas as pd

dummy_data1 = {
        'id': ['1', '2', '3', '4', '5'],
        'Feature1': ['A', 'C', 'E', 'G', 'I'],
        'Feature2': ['B', 'D', 'F', 'H', 'J']}

df1 = pd.DataFrame(dummy_data1, columns = ['id', 'Feature1', 'Feature2'])

#Explore the df
df1

dummy_data2 = {
        'id': ['1', '2', '6', '7', '8'],
        'Feature1': ['K', 'M', 'O', 'Q', 'S'],
        'Feature2': ['L', 'N', 'P', 'R', 'T']}


df2 = pd.DataFrame(dummy_data2, columns = ['id', 'Feature1', 'Feature2'])

df2


dummy_data3 = {
        'id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'Feature3': [12, 13, 14, 15, 16, 17, 15, 12, 13, 23]}

df3 = pd.DataFrame(dummy_data3, columns = ['id', 'Feature3'])

df3

"""
To simply concatenate the DataFrames along the row you can use the concat() function in pandas. 
You will have to pass the names of the DataFrames in a list as the argument to the concat() function:
"""

df_row = pd.concat([df1, df2])

df_row

#correct the row labels

df_row_reindex = pd.concat([df1, df2], ignore_index=True)

df_row_reindex

"""
pandas also provides you with an option to label the DataFrames, after the concatenation, 
with a key so that you may know which data came from which DataFrame. 
You can achieve the same by passing additional argument keys specifying the label names of the DataFrames 
in a list.
"""

frames = [df1,df2]
df_keys = pd.concat(frames, keys=['x', 'y'])

df_keys

#retrieve the data by dataframe

df_keys.loc['y']


#Same result with the dictionary keys

pieces = {'x': df1, 'y': df2}

df_piece = pd.concat(pieces)

df_piece

#To concatenate DataFrames along column, you can specify the axis parameter as 1

df_col = pd.concat([df1,df2], axis=1)

df_col

###############################################################################

"""
Two DataFrames might hold different kinds of information about the same entity 
and linked by some common feature/column. To join these DataFrames, pandas provides 
multiple functions like concat(), merge() , join(), etc.
"""

#Merge

df_merge_col = pd.merge(df_row, df3, on='id')

df_merge_col

"""
id value 1 was present with both A, B and K, L in the DataFrame df_row hence 
this id got repeated twice in the final DataFrame df_merge_col with repeated value 
12 of Feature3 which came from DataFrame df3.
"""
"""
It might happen that the column on which you want to merge the DataFrames have 
different names (unlike in this case). For such merges, you will have to specify 
the arguments left_on as the left DataFrame name and right_on as the right DataFrame name
"""

df_merge_difkey = pd.merge(df_row, df3, left_on='id', right_on='id')

df_merge_difkey

#append rows to a DataFrame by passing a Series or dict to append() function

add_row = pd.Series(['10', 'X1', 'X2', 'X3'],
                    index=['id','Feature1', 'Feature2', 'Feature3'])

df_add_row = df_merge_col.append(add_row, ignore_index=True)

df_add_row

###############################################################################

#Joining SQL style using merge

"""
The FULL OUTER JOIN combines the results of both the left and the right outer joins. 
The joined DataFrame will contain all records from both the DataFrames and fill 
in NaNs for missing matches on either side. You can perform a full outer join 
by specifying the how argument as outer in the merge() function
"""

df_outer = pd.merge(df1, df2, on='id', how='outer')

df_outer

"""
The default suffixes are x and y, however, you can modify them by specifying the 
suffixes argument in the merge() function
"""

df_suffix = pd.merge(df1, df2, left_on='id',right_on='id',how='outer',suffixes=('_left','_right'))

df_suffix

"""
The INNER JOIN produces only the set of records that match in both 
DataFrame A and DataFrame B. You have to pass inner in the how argument 
of merge() function to do inner join
"""

df_inner = pd.merge(df1, df2, on='id', how='inner')

df_inner

"""
The RIGHT JOIN produces a complete set of records from DataFrame B (right DataFrame), 
with the matching records (where available) in DataFrame A (left DataFrame). 
If there is no match, the right side will contain null. 
You have to pass right in the how argument of merge() function to do right join
"""

df_right = pd.merge(df1, df2, on='id', how='right')

df_right

"""
The LEFT JOIN produces a complete set of records from DataFrame A (left DataFrame), 
with the matching records (where available) in DataFrame B (right DataFrame). 
If there is no match, the left side will contain null. 
You have to pass left in the how argument of merge() function to do left join
"""

df_left = pd.merge(df1, df2, on='id', how='left')

df_left

"""
Sometimes you may have to perform the join on the indexes or the row labels. 
To do so, you have to specify right_index (for the indexes of the right DataFrame) 
and left_index (for the indexes of the left DataFrame) as True
"""

df_index = pd.merge(df1, df2, right_index=True, left_index=True)

df_index

###############################################################################

#Challenges
"""
We have a dummy data set from Game of Thrones named house. 
It contains information about various clans (houses). Think of houses as families. 
Let’s say, two new houses have emerged whose information is contained in the data set house_extra.

Task 1: Include the houses in house_extra in the data set house (i.e merging the two data sets)

Suppose you have a new house in the house data frame. 
The new house is “Redwyne” which is present in “The Reach” region. 

Task 2: Add this new observation in our existing house data.

The house data set contains two columns (House, Region). 
Another data set, house_new contains columns (House, Region  and Religion)

Task 3: combine these data sets

Task 4: Get the first row with index 0 from the dataframe resulting from Task 3

Task 5: Solve the issue of index in Task 3

Military strength of each house in military data is given in the same sequence 
as the sequence of houses in house data set.

Task 6: Add military data set to the house data set appropriately

We have a data set candidates which contains information about heirs (successors) 
of each house (family). They are sorted on the basis of age in descending order 
within the same house. (There is no order between the houses)

Task 7: Which of the candidate has the largest army ?

Task 8: List all the houses along with their military strength and the rightful heir

Task 9: List the houses that have atleast one heir

Task 10: List all the available information about houses and heirs

Task 11: Set House as index in both candidates and house dataframes

Task 12: Join the datafames in task 11 by using indexes

Task 13: Differentiate between left and right dataframe columns in the merged dataframe

Task 14: Left merge the dataframes in task 11
"""

###############################################################################
#Let's try it with the tips data

import seaborn as sns
import pandas as pd
import numpy as np

"""
Pandas Combining DataFrames

In pandas there are 4 (plus a few special case) ways to combine data from different frames:

    Merging
    Joining
    Concatenating
    Appending

Where merging and joining are basically redundant and concatenating and appending are basically redundant.

So today we will be going over Merging and Concatenating in pandas.

Check out the full documentation here - http://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
, but be warned it is a bit long :)

Okay let's get started.
"""

tips = sns.load_dataset('tips')
tips.head(3) #Explore the data the way you want

"""
Merge

Merging is for doing complex column-wise combinations of dataframes in a SQL-like way. 
If you don't know SQL joins then check out this resource sql joins.
https://www.w3schools.com/sql/sql_join.asp

To merge we need two dataframes, let's make them below:
"""

tips_bill = tips.groupby(['sex', 'smoker'])[['total_bill', 'tip']].sum()
tips_tip = tips.groupby(['sex', 'smoker'])[['total_bill', 'tip']].sum()

tips_bill
tips_tip


del tips_bill['tip']
del tips_tip['total_bill']

tips_bill
tips_tip

pd.merge?

# we can merge on the indexes
pd.merge(tips_bill, tips_tip, 
         right_index=True, left_index=True)

#we can reset indexes and then merge on the columns - perhaps the easiest way
pd.merge(
    tips_bill.reset_index(), 
    tips_tip.reset_index(),
    on=['sex', 'smoker']
)

# it can actually infer the above - but be very careful with this
pd.merge(
    tips_bill.reset_index(), 
    tips_tip.reset_index()
)

# it can merge on partial column and index
pd.merge(
    tips_bill.reset_index(), 
    tips_tip,
    left_on=['sex', 'smoker'],
    right_index=True
)

#it can do interesting combinations
tips_bill_strange = tips_bill.reset_index(level=0)
tips_bill_strange

pd.merge(
    tips_tip.reset_index(), 
    tips_bill_strange,
    on=['sex', 'smoker']
)

# we can do any SQL-like functionality

#Left
pd.merge(
    tips_bill.reset_index(), 
    tips_tip.reset_index().head(2),
    how='left'
)

#inner
pd.merge(
    tips_bill.reset_index(), 
    tips_tip.reset_index().head(2),
    how='inner'
)

# and if you add an indicator...
pd.merge(
    tips_bill.reset_index().tail(3), 
    tips_tip.reset_index().head(3),
    how='outer',
    indicator=True
)

# it can handle columns with the same name
pd.merge(tips_bill, 
         tips_bill, 
         right_index=True, 
         left_index=True,
         suffixes=('_left', '_right')
)


"""
This is one of the most complex parts of pandas - but it is very important to master. 
So please do check out the excerises below!

One thing to be careful with here is merging two data types. Strings are not equal to ints!

Contatenation

Concatenating is for combining more than two dataframes in either column-wise or row-wise. 
The problem with concatenate is that the combinations it allows you to do are rather simplistic. 
That's why we need merge.

Concatenate can take as many data frames as you want, but it requires that they 
are specifically constructed. All of the dataframes you pass in will need to have 
the same index. So no more using columns as an index.

Let's check out basic use below:
"""

# this adds the dataframes together row wise
pd.concat([tips_bill, tips_bill, tips_tip], sort=False)

# this does it column wise
pd.concat([tips_bill, tips_tip], axis=1)

# and finally this will add on the dataset where it's from
pd.concat([tips_bill, tips_tip], sort=False, keys=['num0', 'num1'])

"""
As you can see there is not a ton of functionality to concat, but it is invaluable 
if you have more than one dataframe or you are looking to append the rows of one dataframe onto another.
"""

###############################################################################

"""
Conclusion

There are a couple of other ways to merge data, but they are pretty niche 
(and mainly for time series data).

They are:

    combine_first
    merge_ordered
    merge_asof

Otherwise you should be fully equipped to do the exercises. 
These functions require a bit of practice to get used to, so don't be discouraged if it takes some time.

"""
###############################################################################