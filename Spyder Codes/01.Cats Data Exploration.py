# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Datasets https://vincentarelbundock.github.io/Rdatasets/datasets.html

import numpy as np

file='https://vincentarelbundock.github.io/Rdatasets/csv/boot/catsM.csv'

cats = np.genfromtxt(file, dtype='float', delimiter=',', 
                     skip_header =1, usecols=[2,3])

print(cats)
"""
Exercises:
    
1. Plot histograms of both body weight and heart weight
2. What do you learn from these histograms?
3. Plot scatter diagram between body weight and heart weight
4. What do you learn from this plot?
5. Model the relation between body weight and heart weight
     considering the heart weight as the response variable
6. Interpret the results of the model fitting
7. Predict the heart weight of the cats having body weight 5 kg, 1.7 kg and 3.155 kg
8. What could be the body weight of the cats having the heart of following weights, 0.025 kg, 0.008 kg, 0.015 kg?
9. Draw a sample of 40 cats, fit the apropriate regression model
10. Compare the parameters of the model derived from 40 cats data with the original data for all the cats.
"""
# 1. Histogram of Body Weight
import matplotlib.pyplot as plt
plt.hist(cats[:,0])
plt.title("Histogram of Body Weight of cats")
plt.xlabel("Body weight in Kg")
plt.ylabel("Frequency")
plt.show()

# 1. Histogram of Heart Weight

plt.hist(cats[:,1])
plt.title("Histogram of Heart Weight of cats")
plt.xlabel("Body weight in Gms")
plt.ylabel("Frequency")
plt.show()

#3. Plot scatter diagram between body weight and heart weight
plt.scatter(cats[:,0]*1000,cats[:,1])
plt.title("Scatter Diagram of Heart Weight vs Body Weight for cats")
plt.xlabel("Body weight in Gms")
plt.ylabel("Heart weight in Gms")
plt.show()