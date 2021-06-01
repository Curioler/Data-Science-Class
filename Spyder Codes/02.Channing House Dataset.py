# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:45:09 2020

@author: Bhavin
"""

import numpy as np


file='https://vincentarelbundock.github.io/Rdatasets/csv/boot/channing.csv'

channing_house = np.genfromtxt(file, dtype='int', delimiter=',', skip_header =1, usecols=[2,3,4,5])
channing_house_gender = np.genfromtxt(file, dtype=str, delimiter=',', skip_header =1, usecols=[1])
print(channing_house)
print(channing_house_gender)

gender = (channing_house_gender == '"Male"')
print(gender)
"""
Exercises:
    
1. What's the minimum, maximum and average entry age?
2. What's the minimum, maximum and average exit age?
3. What's the minimum, maximum and average time duration of stay in the channing house by residents?
4. Plot the histograms of entry, exit and stay duration. What do they show?
5. Plot the boxplots of entry, exit and stay duration. What do they show?
6. How many residets died at the channing house?
7. How any residents are alive or walked out of the channing house alive?
8. What is the minimum duration spent in the channing house where a resident died?
9. Plot two boxplots of entry age, one for the residents who died in the channing house and second for the rest of them? 
10. How many male/female residents stayed in the chaning house?
11. Is there any relationship between gender and entry age?
12. Is there any relationship between gender and deaths in the channing house?
13. Is there any relationship between gender and duration of stay?
14. Find the minimum age of male/female who died in the channing house?
"""

 