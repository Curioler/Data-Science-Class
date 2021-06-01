# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:16:21 2020

@author: Bhavin
"""

###############################################################################
#Some exrcises

"""
Three lists are defined in the script:

    names, containing the country names for which data is available.
    dr, a list with booleans that tells whether people drive left or right in the corresponding country.
    cpc, the number of motor vehicles per 1000 people in the corresponding country.

Each dictionary key is a column label and each value is a list which contains the column elements.
"""

# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict

my_dict = {'country':names, 'drives_right':dr, 'cars_per_cap':cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])

# Print out first 3 observations
print(cars[:3])

# Print out fourth, fifth and sixth observation
print(cars[3:])

# Print out observation for Japan
print(cars.loc['JAP'])

# Print out observations for Australia and Egypt
print(cars.loc[['AUS','EG']])

# Print out drives_right value of Morocco
print(cars.loc['MOR','drives_right'])

# Print sub-DataFrame
print(cars.loc[['RU','MOR'],['country','drives_right']])

# Print out drives_right column as Series
print(cars.loc[:,'drives_right'])

# Print out drives_right column as DataFrame
print(cars.loc[:,['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:,['cars_per_cap','drives_right']])

#Select records using drives_right
sel = cars[cars['drives_right']]

###############################################################################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe

for x, y in europe.items():
    print("the capital of "+x+" is "+y)
    
#Challenge: capitalize the first letter in printing

###############################################################################
    
# Iterate over rows of cars

for lab, row in cars.iterrows():
    print(lab)
    print(row)
    
# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab,'COUNTRY'] = row['country'].upper()

# Print cars
print(cars)

# Use .apply(str.upper)
cars["COUNTRY"] = cars["country"].apply(str.upper)

print(cars)

###############################################################################
"""
Game:
You are walking with your friends in the Empire State Building and you decided to play a game
- Throw a dice 100 times
- If it is 1 or 2, you go one step down
- If it is 3, 4 or 5; you go one step up
- If it is 6, you throw the dice again and walk resulting number of steps up
    i.e., if it is 5 after the 6, you go 6 steps up
- Of course, you cannot go below step 0
- You have 0.1% of chance of falling down the stairs, means you have to start from step 0

Your bet is that you will reach 60 steps high. What is the probability that you will win the bet?

You can solve this problem in two ways:
    Analytically
    Simulating the process

Steps:
The challange for you is to create the simulation
- use random seed 123, dice can be simulated using np.random.randint
- assume that you are at step 50, where would you be after first throw?
- create a randomwalk* starting from 0 using if-elif-else construct
- make sure that the steps can't go below 0
- the last step of the randomwalk is your answer
- plot the randomwalk using line plot
- you can find the distribution of endpoints, i.e. last steps, if you repeat the simulation
- repeat the randomwalks for 10 times and visualize them
- you can do np.random.rand to predict a random number and if it is less than 0.001, 
    you can make the step equal to 0
- visualize the 250 new randomwalks
- create 500 randomwalks, visualize them and find the endpoints of all of them
- plot a histogram of endpoints
- find the probability that you will reach to 60 steps within 100 throws 

*A randomwalk is just a list that shows the step numbers in a sequence based on your random experiment, 
in this case, throwing the dice 
"""
