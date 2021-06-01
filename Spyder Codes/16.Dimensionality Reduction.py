# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:54:32 2020

@author: Bhavin
"""

"""
We are generating a tremendous amount of data daily. In fact, 90% of the data 
in the world has been generated in the last few years! The numbers are truly 
mind boggling. 

Examples:

    Facebook collects data of what you like, share, post, places you visit, restaurants you like, etc.
    Your smartphone apps collect a lot of personal information about you
    Amazon collects data of what you buy, view, click, etc. on their site
    Casinos keep a track of every move each customer makes

I bet you had these questions:
    
- There are too many variables – do I need to explore each and every variable?
- Are all variables important?
- All variables are numeric and what if they have multi-collinearity? 
  How can I identify these variables?
- Is there any machine learning algorithm that can identify the most 
  significant variables automatically?

Having a high number of variables is both a boon and a curse. It’s great that 
we have loads of data for analysis, but it is challenging due to size.

Prime factor - Loss of time

What is Dimensionality Reduction?

In machine learning problems, there are often too many features. 
The higher the number of features, the harder it gets to visualize the 
training set and then work on it. Sometimes, most of these features are 
correlated, and hence redundant.

Dimensionality reduction is the process of reducing the number of random 
variables under consideration, by obtaining a set of principal variables. 
It can be divided into feature selection and feature extraction.

Popular methods used for dimensionality reduction include:

    Principal Component Analysis (PCA) - Linear
    Linear Discriminant Analysis (LDA) - 
    Generalized Discriminant Analysis (GDA)

Benefits:
    Less storage
    Fast computing
    Better model performance
    Removing redundant features

Some thumbrules:
    Drop if more than 40% missing values
    Drop if very low variance, e.g. constants
    Drop one of many features if they all are highly correlated 
        x`- e.g. ‘time spent on treadmill in minutes’ and ‘calories burnt’. 
    Backward elimination by using sum of sqaure of errors
    Forward elimination using performance metrics
    Factor analysis - identifying groups of correlated features

Main ways: feature selection, feature extraction
"""
###############################################################################

"""
Principal Component Analysis

It involves the following steps:

   -Construct the covariance matrix of the data.
   -Compute the eigenvectors of this matrix.
   -Eigenvectors corresponding to the largest eigenvalues are used to 
    reconstruct a large fraction of variance of the original data.

Hence, we are left with a lesser number of eigenvectors, and there might have 
been some data loss in the process. But, the most important variances 
should be retained by the remaining eigenvectors.
"""

#If there are 100 variables in a dataset, how many different scatter graphs can be plotted?

###############################################################################
#Read about the data here: https://www.kaggle.com/uciml/mushroom-classification/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


file = r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\mushrooms.csv'
df = pd.read_csv(file)

df.info()
df.shape
df.head().T

#Converting the classess into integers

encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column])
    
df.head().T

df['cap-color'].max() #10 columns

X = df.iloc[:,1:23]
y = df.iloc[:, 0]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, y, cv=kfold)

#If the warning bugs you, check out this:
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

#Don't forget to check out this - https://www.kaggle.com/vanshjatana/applied-machine-learning
#Thank me later

###############################################################################

#Just selecting two principal components
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['class']]], axis = 1)

finalDf.head()

"""
Note that after dimensionality reduction, there usually isn’t a particular 
meaning assigned to each principal component. The new components are just the 
two main dimensions of variation.
"""
#Explained Variance

"""
The explained variance tells you how much information (variance) can be 
attributed to each of the principal components.
"""
pca.explained_variance_ratio_

###############################################################################
#Without specifying the n_components
pca = PCA()

principalComponents = pca.fit_transform(X)

Percentage_Variance = pca.explained_variance_ratio_

PCA_components = np.arange(1,23)

#Visualizing the variance proportions by PCA
plt.plot(PCA_components, Percentage_Variance, '-bx')
plt.xlabel('PCA Components')
plt.ylabel('Percentage variance')
plt.title('Use elbow method to select the n_components')
plt.show()

#Visualizing the variance proportions by PCA in a bar chart
plt.bar(PCA_components, Percentage_Variance)
plt.xlabel('PCA Components')
plt.ylabel('Percentage variance')
plt.title('Individual variances')
plt.show()

#Adding cumulative percentage
Cumulative_Percent_Variance = Percentage_Variance.cumsum()

plt.plot(PCA_components, Percentage_Variance)
plt.plot(PCA_components, Cumulative_Percent_Variance)
plt.xlabel('PCA Components')
plt.ylabel('Percentage variance')
plt.title('Use elbow method to select the n_components')
plt.xticks(PCA_components)
plt.yticks(np.arange(0,1.1,0.1))
plt.grid()
plt.show()

#Selecting only 12 components
pca = PCA(n_components=12)
principalComponents = pca.fit_transform(X)

#Running K-fold cross validation with principal components
kfold = KFold(n_splits=7, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, principalComponents, y, cv=kfold)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

"""
Let's say you have a straight line 100 yards long and you dropped a penny 
somewhere on it. It wouldn't be too hard to find. You walk along the line and 
it takes two minutes.

Now let's say you have a square 100 yards on each side and you dropped a penny 
somewhere on it. It would be pretty hard, like searching across two football 
fields stuck together. It could take days.

Now a cube 100 yards across. That's like searching a 30-story building the 
size of a football stadium. Ugh.

The difficulty of searching through the space gets a lot harder as you have more dimensions.
"""
###############################################################################

#Carry out the PCA for iris dataset
