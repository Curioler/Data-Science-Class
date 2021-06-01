# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:23:50 2020

@author: Bhavin
"""
###############################################################################
#Simple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\salary_data.csv')

#Explore the dataset
"""
Interests:
    How many variables?
    How many observations?
    Any missing value?
    Any outliers?
    Datatypes?
    Distributions?
    Relations?
    Where is your response variable?
"""
X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
y = dataset.iloc[:, 1].values #get array of dataset in  1st column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualizing the Training set results

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Predicting the result of 5 Years Experience
y_pred = regressor.predict(np.array(0).reshape(-1,1))
y_pred

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Checking the accuracy
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#R-squared score

print('R-squared score:', regressor.score(X_test, y_test))  


###############################################################################
#Multiple linear regression

winequalityfile = r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\winequality-red.csv'

dataset = pd.read_csv(winequalityfile)

#Explore your data

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#Exploring the effets of all the variables
variables = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']

coeff_df = pd.DataFrame(regressor.coef_, index=variables, columns=['Coefficient'])  
coeff_df

#Predicting the test set values
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred,0)}, dtype=int)
df.head()

#Performance evaluation

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R-squared score:', regressor.score(X_test, y_test))  

# Prepare classification report and confusion matrix

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

confusion_matrix(df.Actual,df.Predicted)
iteration1 = accuracy_score(df.Actual,df.Predicted)
#classification_report(df.Actual,df.Predicted)

#Which is the best approach? Classification or regression?

#Remove any two variables who you think are not very useful in the algorithm and train the algorithm again

###############################################################################
#Iteration 2

#EDA
dataset.info()
dataset.head().T

#Checking class imbalance
dataset.groupby('quality').size()

#Let's see some graphs
import seaborn as sns

#fixed acidity
sns.barplot(x = 'quality', y = 'fixed acidity', data = dataset) #does not look helpful

#volatile acidity
sns.barplot(x = 'quality', y = 'volatile acidity', data = dataset) # quality increases as volatile acidity decreases

#citric acid
sns.barplot(x = 'quality', y = 'citric acid', data = dataset) #quality increases as citric acid increases

#residual sugar
sns.barplot(x = 'quality', y = 'residual sugar', data = dataset) #does not look helpful

#chlorides
sns.barplot(x = 'quality', y = 'chlorides', data = dataset) #quality increases as chlorides decreases

#free sulfur dioxide
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = dataset) #indecisive

#total sulfur dioxide
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = dataset) #indecisive

#density
sns.barplot(x = 'quality', y = 'density', data = dataset) #does not look helpful

#pH
sns.barplot(x = 'quality', y = 'pH', data = dataset) #does not look helpful

#sulphates
sns.barplot(x = 'quality', y = 'sulphates', data = dataset) #quality increases as sulphates increases

#alcohol
sns.barplot(x = 'quality', y = 'alcohol', data = dataset) #quality increases as alcohol increases

#chec out https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine

#Let's drop the not useful variables

dataset1 = dataset.drop(['fixed acidity','residual sugar','density','pH'], axis=1)
dataset1.columns

#splitting dataset into features and target
X = dataset1.iloc[:,:-1].values
y = dataset1.iloc[:,-1].values

#standardizing the data
from sklearn.preprocessing import StandardScaler

#check https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#Exploring the effets of all the variables
variables = ['volatile acidity', 'citric acid',
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates','alcohol']

coeff_df = pd.DataFrame(regressor.coef_, index=variables, columns=['Coefficient'])  
coeff_df #compare it with the above graphs

#Predicting the test set values
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred,0)}, dtype=int)
df.head()
df.tail()

#Performance evaluation
print('R-squared score:', regressor.score(X_test, y_test))  

confusion_matrix(df.Actual,df.Predicted)
iteration2 = accuracy_score(df.Actual,df.Predicted)
#classification_report(df.Actual,df.Predicted)
#Do it with a classification algorithm, does it help?

###############################################################################
#Iteration 3

#let's drop the indecisive ones too

dataset2 = dataset1.drop(['free sulfur dioxide','total sulfur dioxide'], axis=1)
dataset2.columns

X = dataset2.iloc[:,:-1].values
y = dataset2.iloc[:,-1].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#Exploring the effets of all the variables
variables = ['volatile acidity', 'citric acid', 'chlorides', 'sulphates','alcohol']

coeff_df = pd.DataFrame(regressor.coef_, index=variables, columns=['Coefficient'])  
coeff_df #compare it with the above graphs

#Predicting the test set values
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred,0)}, dtype=int)
df.head()
df.tail()

#Performance evaluation
print('R-squared score:', regressor.score(X_test, y_test))  

confusion_matrix(df.Actual,df.Predicted)
iteration3 = accuracy_score(df.Actual,df.Predicted)
#classification_report(df.Actual,df.Predicted)

###############################################################################
print(round(iteration1,2), round(iteration2,2), round(iteration3,2))

###############################################################################
"""
Get the data from the following link and carry out linear regression
https://www.kaggle.com/divan0/multiple-linear-regression/data

It's a competition to get the best R2 score
"""