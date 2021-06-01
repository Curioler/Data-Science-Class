# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:47:52 2020

@author: Bhavin
"""

"""
A time series is any data set where the values are measured at different 
points in time.

Many time series are uniformly spaced at a specific frequency, for example, 
hourly weather measurements, daily counts of web site visits, or monthly 
sales totals.

See the following link for more details on our use case:

https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/

We will explore how electricity consumption and production in Germany have 
varied over time, using pandas time series tools to answer questions such as:

   -When is electricity consumption typically highest and lowest?
   -How do wind and solar power production vary with seasons of the year?
   -What are the long-term trends in electricity consumption, solar power, and
    wind power?
   -How do wind and solar power production compare with electricity consumption,
    and how has this ratio changed over time?
"""
###############################################################################
#Revision of pandas datetime

import pandas as pd
pd.to_datetime('2018-01-15 3:45pm')

pd.to_datetime('7/8/1952')

pd.to_datetime('7/8/1952', dayfirst=True)

pd.to_datetime(['2018-01-05', '7/8/1952', 'Oct 10, 1995'])

pd.to_datetime(['2/25/10', '8/6/17', '12/15/12'], format='%m/%d/%y')

"""
To work with time series data in pandas, we use a DatetimeIndex as the index 
for our DataFrame (or Series).
"""
###############################################################################
data = r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\opsd_germany_daily.csv'

df = pd.read_csv(data)
df.shape

df.head(3)
df.tail(3)

df.dtypes

df.info()

#Reading the data with date in datetime format
df = pd.read_csv(data, parse_dates=['Date'])

#Setting datetime as index
df.set_index('Date', inplace=True)

df.tail(3)
df.info()

df.index

#Reading the data with date as index and in date format
df = pd.read_csv(data, index_col=0, parse_dates=True)

###############################################################################
# Add columns with year, month, and weekday name
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Weekday Name'] = df.index.weekday_name

# Display a random sampling of 5 rows
df.sample(5, random_state=0)

#Data for a single day
df.loc['2017-08-10']

#Slice of dates
df.loc['2014-01-20':'2014-01-22']

#Partial string indexing
df.loc['2012-02']

df.loc['2013']

###############################################################################
#Visualizing time series data

import matplotlib.pyplot as plt

import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

df['Consumption'].plot(linewidth=0.5) #Noe the xticks

cols_plot = ['Consumption', 'Solar', 'Wind']
axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None',
         figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')
plt.show()

#For the years 2012 to 2014
axes = df.loc['2012':'2014',cols_plot].plot(marker='.', alpha=0.5, 
             linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')
plt.show()
#When is the electricity consumption high? low? why?
#Any different patterns in consumption?
#When is solar power production highest? lowest? why?
#When is wind power production highest? lowest? why?
#What can you tell about the trend of production?    

"""
All three time series clearly exhibit periodicity—often referred to as 
seasonality —in which a pattern repeats again and again at regular time 
intervals.

The Consumption, Solar, and Wind time series oscillate between high and low 
values on a yearly time scale, corresponding with the seasonal changes in 
weather over the year. However, seasonality in general does not have to 
correspond with the meteorological seasons. For example, retail sales data 
often exhibits yearly seasonality with increased sales in November and December,
due to the holidays.
"""

#Let’s plot the time series in a single year to investigate further.

ax = df.loc['2017', 'Consumption'].plot()
ax.set_ylabel('Daily Consumption (GWh)')
plt.show()

#Why do you think the oscilations are happening? Anything else interesting?

#Let’s zoom in further and look at just January and February.

ax = df.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')
ax.set_ylabel('Daily Consumption (GWh)')
plt.show()

#With weekday names
ax = df.loc['2017-01-01':'2017-01-22', 'Consumption'].plot(marker='o', linestyle='-')

tickvalues = df.loc['2017-01-01':'2017-01-22', 'Consumption'].index
days = df.loc['2017-01-01':'2017-01-22', 'Weekday Name'].tolist()
plt.xticks(ticks = tickvalues ,labels = days, rotation = 'vertical')

ax.set_ylabel('Daily Consumption (GWh)')
ax.set_title('Consumption')
plt.show()

###############################################################################
"""
There are many other ways to visualize time series, depending on what patterns
you’re trying to explore
"""

#Seasonality

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
    sns.boxplot(data=df, x='Month', y=name, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_title(name)

# Remove the automatic x-axis label from all but the bottom subplot
if ax != axes[-1]:
    ax.set_xlabel('')
plt.show()
    
"""
Let’s group the electricity consumption time series by day of the week, to 
explore weekly seasonality
"""
sns.boxplot(data=df, x='Weekday Name', y='Consumption')
plt.show()

#What can you say about outliers?

###############################################################################
#Resampling

"""
We use the DataFrame’s resample() method, which splits the DatetimeIndex into 
time bins and groups the data by time bin. The resample() method returns a 
Resampler object, similar to a pandas GroupBy object. We can then apply an 
aggregation method such as mean(), median(), sum(), etc., to the data group 
for each time bin.
"""

# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
# Resample to weekly frequency, aggregating with mean
weekly_mean = df[data_columns].resample('W').mean()
weekly_mean.head(3) #First row is mean of 2006-01-01 through 2006-01-07

print(df.shape[0])
print(weekly_mean.shape[0])

"""
Let's Plot the daily and weekly Solar time series together over a single six-month 
period to compare them.
"""

# Start and end of the date range to extract
start, end = '2017-01', '2017-06'

# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Solar'],
        marker='.', linestyle='-', linewidth=0.5, label='Daily')

ax.plot(weekly_mean.loc[start:end, 'Solar'],
        marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')

ax.set_ylabel('Solar Production (GWh)')
ax.legend()
plt.show()

#What can you tell about the weekly data?

"""
Compute the monthly sums, setting the value to NaN for any month which has
fewer than 28 days of data
"""

monthly = df[data_columns].resample('M').sum(min_count=28)
monthly.head(3) #MOnthly data are labeled with the last day of the month

#By default, right bin edge for monthly, quarterly, and annual frequencies
#Left bin edge for all other frequencies

#See the resample() documentation to check the parameters and how to change them

#monthly - line plot

fig, ax = plt.subplots()
ax.plot(monthly['Consumption'], color='black', label='Consumption')
ax.plot(monthly['Wind'], color='red', label='Wind')
ax.plot(monthly['Solar'], color='blue', label='Solar')
ax.legend()
ax.set_ylabel('Monthly Total (GWh)')
plt.show()

"""
Compute the annual sums, setting the value to NaN for any year which has
fewer than 360 days of data
"""
annual = df[data_columns].resample('A').sum(min_count=360)

"""
The default index of the resampled DataFrame is the last day of each year,
('2006-12-31', '2007-12-31', etc.) so to make life easier, set the index
to the year component
"""

annual.set_index(annual.index.year, inplace=True)
annual

annual.index.name = 'Year'

# Compute the ratio of Wind+Solar to Consumption
annual['Wind+Solar/Consumption'] = annual['Wind+Solar'] / annual['Consumption']
annual.tail(3)

# Plot from 2012 onwards, because there is no solar production data in earlier years
ax = annual.loc[2012:, 'Wind+Solar/Consumption'].plot.bar()
ax.set_ylabel('Fraction')
ax.set_ylim(0, 0.3)
ax.set_title('Wind + Solar Share of Annual Electricity Consumption')
plt.xticks(rotation=0)
plt.show()

###############################################################################
#Rolling window

"""
Similar to downsampling, rolling windows split the data into time windows and 
and the data in each window is aggregated with a function such as mean(),
median(), sum(), etc. However, unlike downsampling, where the time bins do not 
overlap and the output is at a lower frequency than the input, rolling windows 
overlap and “roll” along at the same frequency as the data, so the transformed 
time series is at the same frequency as the original time series.
"""

# Compute the centered 7-day rolling mean
rolling7d = df[data_columns].rolling(7, center=True).mean()
rolling7d.head(10) #center=True argument to label each window at its midpoint
rolling7d.tail(10)

#2006-01-01 to 2006-01-07 — labelled as 2006-01-04

# Start and end of the date range to extract
start, end = '2017-01', '2017-06'
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Solar'], color='green',
        marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(weekly_mean.loc[start:end, 'Solar'], color = 'blue',
        marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.plot(rolling7d.loc[start:end, 'Solar'], color = 'black',
        marker='.', linestyle='-', label='7-d Rolling Mean')
ax.set_ylabel('Solar Production (GWh)')
ax.legend()
plt.show()


"""
We can see that data points in the rolling mean time series have the same 
spacing as the daily data, but the curve is smoother because higher frequency 
variability has been averaged out.
"""
###############################################################################
#Trend

"""
An easy way to visualize these trends is with rolling means at different time 
scales. Rolling means tend to smooth a time series by averaging out variations 
at frequencies much higher than the window size and averaging out any 
seasonality on a time scale equal to the window size.
"""

#Electricity consumption has weekly and yearly seasonality,
#let’s look at rolling means on those two time scales

# The min_periods=360 argument accounts for a few isolated missing days in the
# wind and solar production time series
rolling365d = df[data_columns].rolling(window=365, center=True, min_periods=360).mean()

# Plot daily, 7-day rolling mean, and 365-day rolling mean time series
fig, ax = plt.subplots()
ax.plot(df['Consumption'], marker='.', markersize=2, color='0.6',
        linestyle='None', label='Daily')
ax.plot(rolling7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
ax.plot(rolling365d['Consumption'], color='0.2', linewidth=3,
        label='Trend (365-d Rolling Mean)')
# Set x-ticks to yearly interval and add legend and labels
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')
ax.set_title('Trends in Electricity Consumption')
plt.show()

#7-day rolling mean has smoothed out all the weekly seasonality, 
#while preserving the yearly seasonality

#What can you say about electricity consumption looking at the rolling 365 day trend?

# Plot 365-day rolling mean time series of wind and solar power
fig, ax = plt.subplots()
for nm in ['Wind', 'Solar', 'Wind+Solar']:
    ax.plot(rolling365d[nm], label=nm)
    # Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
    ax.set_ylim(0, 400)
    ax.legend()
    ax.set_ylabel('Production (GWh)')
    ax.set_title('Trends in Electricity Production (365-d Rolling Means)')
plt.show()

###############################################################################

#Hypothesis Testing for the presence of trend

from statsmodels.tsa.stattools import adfuller

def trendcheck(timeseries):
    ts = timeseries.dropna()
    adfullertest = adfuller(ts, autolag='AIC')
    p_value = adfullertest[1]

    if p_value < 0.05:
        test_result = 'Stationary timeseries'
    else:
        test_result = "Non-stationary timeseries"
    return test_result

for timeseries in data_columns:
    print(timeseries,":",trendcheck(df[timeseries]))
    
for timeseries in data_columns:
    print(timeseries,":",trendcheck(rolling7d[timeseries]))

#Stationary - No significant trend/seasonality
#Read about adfuller test - https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

###############################################################################
    
#Forecasting

#Check out different techniques - 
#https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/

#How to make a timeseries stationary?
#Answer: By taking a difference of original and moving averages timeseries

    
solar = df['Solar'] - rolling7d['Solar']
plt.plot(solar, label='After taking a difference')
plt.plot(df['Solar'], label='Original')
plt.title('Comparison of before and after taking difference')
plt.legend()
plt.show()

#Let's do the test for stationarity
trendcheck(solar)

#Differencing: to remove trend and seasonality

solar = df['Solar'] - df['Solar'].shift()
plt.plot(solar, label='After taking a difference')
plt.plot(df['Solar'], label='Original')
plt.title('Comparison of before and after taking difference')
plt.legend()
plt.show()

#Let's do the test for stationarity
trendcheck(solar)

###############################################################################
#Decomposing

"""
A useful abstraction for selecting forecasting methods is to break a time 
series down into systematic and unsystematic components.

    Systematic: Components of the time series that have consistency or 
                recurrence and can be described and modeled.
    Non-Systematic: Components of the time series that cannot be directly modeled.

A given time series is thought to consist of three systematic components 
including level, trend, seasonality, and one non-systematic component called noise.

These components are defined as follows:

    Level: The average value in the series.
    Trend: The increasing or decreasing value in the series.
    Seasonality: The repeating short-term cycle in the series.
    Noise: The random variation in the series.

All series have a level and noise. The trend and seasonality components are optional.

Check out additive and multiplicative models at the below link:
    
https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
"""
from statsmodels.tsa.seasonal import seasonal_decompose

#An example of a madeup series
from random import randrange

series = [i+randrange(10) for i in range(1,100)]
result = seasonal_decompose(series, model='additive', freq=1)
result.plot()

#Let's decompose the solar series
series = df['Solar'].dropna().values
result = seasonal_decompose(series, model='additive', freq=1) #Naive decompose method
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)

#Visualize them
result.plot()
plt.show()

#Let's get a new dataset
newfile = r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\airline-passengers.csv'

al = pd.read_csv(newfile, index_col=0, parse_dates=['Month'])
al.info()

al.plot()
plt.show()

"""
There may be a linear trend
There is also seasonality, but the amplitude (height) of the cycles appears to 
    be increasing, suggesting that it is multiplicative
"""

#Decomposing
result = seasonal_decompose(al, model='multiplicative')
result.plot()
plt.show()


#I will leave the exploring part to you people

#Test for trend and seasonality
trendcheck(al['Passengers'])

#Transformation
import numpy as np

ts = np.log(al['Passengers'])

"""
One of the first tricks to reduce trend can be transformation. For example, 
in this case we can clearly see that the there is a significant positive trend. 
So we can apply transformation which penalize higher values more than smaller 
values. These can be taking a log, square root, cube root, etc. Lets take a log 
transform here for simplicity
"""
plt.plot(ts)

#Decomposing
result = seasonal_decompose(ts, model='additive')
result.plot()
plt.show()

#Differencing - to remove the trend

ts_diff = ts - ts.shift()
plt.plot(ts_diff)
plt.show()
