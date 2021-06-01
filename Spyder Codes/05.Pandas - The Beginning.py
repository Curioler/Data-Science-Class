# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:25:01 2020

@author: Bhavin
"""

"""
###############################################################################
- Pandas is the backbone of most data projects
- [pandas] is derived from the term "panel data", 
  an econometrics term for data sets that include observations over multiple 
  time periods for the same individuals.
- Core components of pandas: Series, DataFrames and Panels
- The primary two components of pandas are the Series and DataFrame.
  A Series is essentially a column, and a DataFrame is a multi-dimensional table
  made up of a collection of Series.

###############################################################################
"""


#Import pandas
import pandas as pd

#Create a dictionary
data = {
    'apples': [3, 2, 0, 1], 
    'oranges': [0, 3, 7, 2]
}

#Dataframe constructor
purchases = pd.DataFrame(data)

print(purchases)

"""The Index of this DataFrame was given to us on creation as the numbers 0-3,
 but we could also create our own when we initialize the DataFrame."""
 
purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])

print(purchases)

#locate a customer's order by using their name

purchases.loc['June']

#Reading data from CSVs

file = "https://vincentarelbundock.github.io/Rdatasets/csv/boot/acme.csv"

df = pd.read_csv(file)

print(df)

"""CSVs don't have indexes like our DataFrames, so all we need to do is 
just designate the index_col when reading."""

df = pd.read_csv(file, index_col=0)

print(df)


#Reading data from JSON

json_string = '{"apples": [3, 2, 0, 1], "oranges": [0, 3, 7, 2], "bananas":[5,0,0,3]}'
    #Note the quotations

df = pd.read_json(json_string)

print(df)

###############################################################################
"""
Difference between json and dict
- JSON is a pure string and dict is a data structure
- dictionary key can be any hash object, and JSON can only be a string
- dict string uses single quotation marks, and JSON enforces double quotation marks
"""
###############################################################################

#Converting back to a CSV or JSON

df.to_csv('new_purchases.csv')

df.to_json('new_purchases.json') #check the index and set it right then read it back


###############################################################################
#Most important DataFrame operations

market_vs_acme = pd.read_csv(file, index_col="month")
print(market_vs_acme)

#Viewing your data

breslow = "https://vincentarelbundock.github.io/Rdatasets/csv/boot/breslow.csv"

df = pd.read_csv(breslow, index_col=0)

df.head()
df.tail()


#Getting info about your data

df.info() #should be your very first command after loading your data

#Anothe useful attribute
df.shape

#Handling duplicates

temp_df = df.append(df)
temp_df.shape

print(temp_df) #check the index 

"""
Using append() will return a copy without affecting the original DataFrame.
We are capturing this copy in temp so we aren't working with the real data.
"""
temp_df = temp_df.drop_duplicates()

temp_df.shape

"""
Just like append(), the drop_duplicates() method will also return a copy of your DataFrame, 
but this time with duplicates removed.
"""

#inplace
"""
It's a little verbose to keep assigning DataFrames to the same variable like in this example. 
For this reason, pandas has the inplace keyword argument on many of its methods. 
Using inplace=True will modify the DataFrame object in place:
"""

temp_df = df.append(df)
temp_df.shape

temp_df.drop_duplicates(inplace=True)
temp_df.shape

#Keep option for drop_duplicates()

"""
Another important argument for drop_duplicates() is keep, which has three possible options:

    first: (default) Drop duplicates except for the first occurrence.
    last: Drop duplicates except for the last occurrence.
    False: Drop all duplicates.

Since we didn't define the keep arugment in the previous example it was defaulted to first.
This means that if two rows are the same pandas will drop the second row and keep the first row.
Using last has the opposite effect: the first row is dropped.

keep, on the other hand, will drop all duplicates.
If two rows are the same then both will be dropped.
Watch what happens to temp_df:
"""

temp_df = df.append(df)
temp_df.shape

temp_df.drop_duplicates(inplace=True, keep=False)
temp_df.shape

#print the column names of dataframe

df.columns

#Renaming column names

df.rename(columns={
        'n': 'Person Years', 
        'y': 'Deaths by CAD',
        'ns': 'Smoker Years'
    }, inplace=True)

df.columns

#But best naming would be

df.rename(columns={
        'Person Years': 'person_years', 
        'Deaths by CAD': 'deaths_cad',
        'Smoker Years': 'smoker_years'
    }, inplace=True)

df.columns

#Rename columns by assigning

df.columns = ["Age", "Smoke", "Person_years", "Deaths_cad", "Smoker_years"]
df.columns

#Using list comprehension :-O

df.columns = [col.lower() for col in df.columns] #Don't be scared, we will do it in details

df.columns

#list (and dict) comprehensions come in handy a lot when working with pandas and data in general.

"""
It's a good idea to lowercase, remove special characters, and replace spaces with underscores 
if you'll be working with a dataset for some time.
"""

###############################################################################

"""
How to work with missing values

When exploring data, you’ll most likely encounter missing or null values, 
which are essentially placeholders for non-existent values. 
Most commonly you'll see Python's None or NumPy's np.nan, 
each of which are handled differently in some situations.

There are two options in dealing with nulls:

    Get rid of rows or columns with nulls
    Replace nulls with non-null values, a technique known as imputation

Let's calculate to total number of nulls in each column of our dataset. 
The first step is to check which cells in our DataFrame are null:


"""
imdb_data = r"C:\Users\Bhavin\Google Drive\Data Science Class\Assets\IMDB-Movie-Data.csv"
movies_df = pd.read_csv(imdb_data)

movies_df.info()

movies_df = pd.read_csv(imdb_data,index_col="Title")

movies_df.columns = ['rank', 'genre', 'description', 'director', 'actors', 'year', 'runtime', 
                     'rating', 'votes', 'revenue_millions', 'metascore']
movies_df.info()
movies_df.head()
movies_df.tail()

#To check which cells in our DataFrame are null
movies_df.isnull()

#To count the number of nulls in each column we use an aggregate function for summing
movies_df.isnull().sum()

#removing nulls
movies_df.dropna() #delete any row with at least a single null value, use inplace=True

#Other than just dropping rows, you can also drop columns with null values by setting axis=1
movies_df.dropna(axis=1) #use inplace=True

###############################################################################
"""
What's with this axis=1parameter?

It's not immediately obvious where axis comes from and why you need it to be 1 for it to affect columns. 
To see why, just look at the .shape output:

movies_df.shape

Out: (1000, 11)

As we learned above, this is a tuple that represents the shape of the DataFrame, 
i.e. 1000 rows and 11 columns. 
Note that the rows are at index zero of this tuple and columns are at index one of this tuple. 
This is why axis=1 affects columns. 
This comes from NumPy, and is a great example of why learning NumPy is worth your time.
"""

"""
Imputation

Imputation is a conventional feature engineering technique used to keep valuable data that have null values.

There may be instances where dropping every row with a null value removes too big a chunk from your dataset, 
so instead we can impute that null with another value, usually the mean or the median of that column.

"""
###############################################################################

revenue = movies_df['revenue_millions']
type(revenue)

revenue.head() #see the index
revenue.tail()
#revenue.info() #what do you think will happen?

#mean revenue
revenue_mean = revenue.mean()

revenue_mean

#imputing the missing values
revenue.fillna(revenue_mean, inplace=True) 

revenue.tail()

#Notice that by using inplace=True we have actually affected the original movies_df
movies_df['revenue_millions']
movies_df.isnull().sum()

"""
Imputing an entire column with the same value like this is a basic example. 
It would be a better idea to try a more granular imputation by Genre or Director.

For example, you would find the mean of the revenue generated in each genre individually 
and impute the nulls in each genre with that genre's mean.
"""

#Summary of the distribution of continuous variables:
movies_df.describe()
type(movies_df.describe())

"""
.describe() can also be used on a categorical variable to get the count of rows,
 unique count of categories, top category, and freq of top category
"""
movies_df['genre'].describe()

type(movies_df['genre'].describe())

#.value_counts() can tell us the frequency of all values in a column

movies_df['genre'].value_counts().head(10)


###############################################################################

#Relationships between continuous variables

movies_df.corr() #look at the value of votes vs revenue
type(movies_df.corr())

"""
Examining bivariate relationships comes in handy when you have an outcome 
or dependent variable in mind and would like to see the features most correlated 
to the increase or decrease of the outcome. 
You can visually represent bivariate relationships with scatterplots
"""
###############################################################################

#Slicing, selecting, extracting

#Series vs DataFrame

genre_col = movies_df['genre']

type(genre_col)

genre_col.info() #Gives an error

#selecting two columns
subset = movies_df[['genre', 'rating']]

subset.head()

"""
For selecting rows, we have two options:

    .loc - locates by name
    .iloc- locates by numerical index
"""
#Selecting one record
prom = movies_df.loc["Prometheus"]

prom

prom = movies_df.iloc[1]

prom


#Selecting multiple records
movie_subset = movies_df.loc['Prometheus':'Sing']

movie_subset

movie_subset = movies_df.iloc[1:4]

movie_subset

"""
One important distinction between using .loc and .iloc to select multiple rows is that
 .loc includes the movie Sing in the result, but when using .iloc we're getting rows 1:4 
 but the movie at index 4 (Suicide Squad) is not included.
"""

#Filtering: Take a column from the DataFrame and apply a Boolean condition to it

condition = (movies_df['director'] == "Ridley Scott")

condition.head()

movies_df[condition]

movies_df[movies_df['director'] == "Ridley Scott"]

#One more filtering
movies_df[movies_df['rating'] >= 8.6].head(3)

#Richer conditions: Logical operators | for "or" and & for "and"
movies_df[(movies_df['director'] == 'Christopher Nolan') | (movies_df['director'] == 'Ridley Scott')].head()
 #Note the use of brackets

#Another way of doing it
movies_df[movies_df['director'].isin(['Christopher Nolan', 'Ridley Scott'])].head()


###############################################################################
"""
Challenge:
Find all movies that 
- were released between 2005 and 2010, 
- have a rating above 8.0, 
- but made below the 25th percentile in revenue (use .quantile(0.25) if you want to)   
"""
###############################################################################

"""
Applying functions

It is possible to iterate over a DataFrame or Series as you would with a list, 
but doing so — especially on large datasets — is very slow.

An efficient alternative is to apply() a function to the dataset.
"""

"""
Use a function to convert movies with an 8.0 or greater to a string value of "good" 
and the rest to "bad" and use this transformed values to create a new column.
"""
#Defining the function
def rating_function(x):
    if x >= 8.0:
        return "good"
    else:
        return "bad"
    
#Send the entire rating column through this function
movies_df["rating_category"] = movies_df["rating"].apply(rating_function)

#Some coll options

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.set_option('display.precision', 2)

movies_df.head(1)

#Homework: Find out how to reset these options


#Using lambda function: Recall lambda arg: expression 
movies_df["rating_category"] = movies_df["rating"].apply(lambda x: 'good' if x >= 8.0 else 'bad')

movies_df.head(2)

"""
Using apply() will be much faster than iterating manually over rows because 
pandas is utilizing vectorization
"""

###############################################################################
#Plotting

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)}) # set font and plot size to be larger

#Scatter Diagram
movies_df.plot(kind='scatter', x='rating', y='revenue_millions', title='Revenue (millions) vs Rating');

#Check out https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html

"""
What's with the semicolon? Just a way to hide 
the <matplotlib.axes._subplots.AxesSubplot at 0x26613b5cc18> output.
"""
#Does the combination of statements work?
movies_df.plot(kind='scatter', x='rating', y='revenue_millions', title='Revenue (millions) vs Rating')
plt.xlabel('Ratings') 
plt.ylabel('Revenue in mn') 
plt.show()

#Histogram
movies_df['rating'].plot(kind='hist', title='Rating')

movies_df['rating'].plot(kind='hist', title='Rating',color='green') #The options work

#Boxplot
movies_df['rating'].plot(kind="box")

#Group of box plots
movies_df.boxplot(column='revenue_millions', by='rating_category')

###############################################################################
"""
Challange: Plot all the above graphs only using matplotlib
"""