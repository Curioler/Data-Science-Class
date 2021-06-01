# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 08:04:45 2020

@author: Bhavin
"""

"""
1. Download the titanic dataset from kaggle challange
2. Read the data in python
3. Explore the data
4. Find out which variables affect the survival
5. Train a classification algorithm
6. Find the accurary of the algorithm
"""
###############################################################################
#1. Download the titanic dataset from kaggle challange
"""
- Read the challange description
- Read the data dictionary
"""
file = r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\Titanic dataset\train.csv'

###############################################################################
#2. Read the data in python
"""
- Read the data in python
"""
import pandas as pd

df = pd.read_csv(file)

###############################################################################
#3. Explore the data
"""
- Explore the data - size, datatype
- Check missing values, unusual values
- Check associations
"""

df.info()
df.shape
df.head()
df.head(2).T
df.tail()
df.tail(2).T
df.describe()
df.describe().T
df.columns
df.loc[:,['Survived','Pclass','Age','SibSp','Parch','Fare']].describe()
round(df.loc[:,['Survived','Pclass','Age','SibSp','Parch','Fare']].describe(),2)
df[['Sex','Cabin', 'Embarked']].describe()
df.groupby('Sex')['PassengerId'].count()
df.groupby('Embarked')['PassengerId'].count()
round(df[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr(),2)

df_wo_na = df.loc[:,['Survived','Pclass','Age','SibSp','Parch','Fare']].dropna()

import seaborn as sns
sns.set(style="ticks", color_codes=True)

sns.pairplot(df_wo_na)

sns.pairplot(df_wo_na, hue='Survived')

###############################################################################
#4. Find out which variables affect the survival
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
n = 3

survival = df.groupby(['Survived','Pclass'])['PassengerId'].count()
survival

survived = survival.loc[1]
not_survived = survival.loc[0]

survived
not_survived

fig, ax = plt.subplots()

index = np.arange(n)
bar_width = 0.35
opacity = 0.9

ax.bar(index, survived, bar_width, alpha=opacity, color='r',
                label='Survived')
ax.bar(index+bar_width, not_survived, bar_width, alpha=opacity, color='b',
                label='Not survived')

ax.set_xlabel('Passanger Class')
ax.set_ylabel('# of passengers')
ax.set_title('Survived v/s Not survived')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('First Class','Second Class','Third Class'))

ax.legend()

plt.show()

# Stacked Chart
r = [0,1,2]

# From raw value to percentage
totals = [i+j for i,j in zip(survived, not_survived)]
greenBars = [(i / j) * 100 for i,j in zip(survived, totals)]
orangeBars = [(i / j) * 100 for i,j in zip(not_survived, totals)]
 
# plot
barWidth = 0.85
names = ('First Class','Second Class','Third Class')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label='Survived')
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label='Not survived')
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Passnger Class")

plt.legend()
# Show graphic
plt.show()

#Survival By Age and Sex

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = df[df['Sex']=='female']
men = df[df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), 
                  bins=18, 
                  label = 'Survived', 
                  ax = axes[0], 
                  kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), 
                  bins=40, 
                  label = 'Not survived', 
                  ax = axes[0], 
                  kde =False)

ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), 
                  bins=18, 
                  label = 'Survived', 
                  ax = axes[1], 
                  kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), 
                  bins=40, 
                  label = 'Not survived', 
                  ax = axes[1], 
                  kde = False)
ax.legend()
ax.set_title('Male')

"""
Men have a high probability of survival when they are between 18 and 30 years old
For women the survival chances are higher between 14 and 40
For men the probability of survival is very low between the age of 5 and 18
Infants also have a little bit higher probability of survival
"""
#Certain groups have higher probability of survival
#We may want to create the age groups

df.groupby(['Survived','Embarked','Sex'])['PassengerId'].count()
survived = df.groupby(['Survived','Embarked','Sex'])['PassengerId'].count()[1]
not_survived = df.groupby(['Survived','Embarked','Sex'])['PassengerId'].count()[0]
embarked_sex = pd.concat([survived, not_survived], axis=1, keys=['Survived','Not Survived'])
embarked_sex['Total'] = embarked_sex['Survived']+embarked_sex['Not Survived']
embarked_sex['Survived'] = round(embarked_sex['Survived'] / embarked_sex['Total'],3)  
embarked_sex['Not Survived'] = round(embarked_sex['Not Survived']/embarked_sex['Total'],3)

embarked_sex
embarked_sex['Survived'] = embarked_sex['Survived'] * 100
embarked_sex['Not Survived'] = embarked_sex['Not Survived'] * 100
del embarked_sex['Total']
embarked_sex

##########################################

#Checking missing values
df.isnull().sum().sort_values(ascending=False)

round(df.isnull().sum()/df.isnull().count()*100,1).sort_values(ascending=False)

total = df.isnull().sum().sort_values(ascending=False)
percent = round(df.isnull().sum()/df.isnull().count()*100,1).sort_values(ascending=False)

missing_values = pd.concat([total,percent], axis=1, keys=['Total','%'])
missing_values
missing_values.head(4)

"""
Cabin: 77% missing values, needs further investigation, might drop the column
Age: 177 values, very tricky
Embarked: 2 values, can be easilly filled
"""

###############################################################################

# Stacked Chart for embarked

survival_embarked = df.groupby(['Survived','Embarked'])['PassengerId'].count()
survival_embarked

survived_embarked = survival_embarked.loc[1]
not_survived__embarked = survival_embarked.loc[0]

survived_embarked
not_survived__embarked


r = [0,1,2]

# From raw value to percentage
totals = [i+j for i,j in zip(survived_embarked, not_survived__embarked)]
greenBars = [(i / j) * 100 for i,j in zip(survived_embarked, totals)]
orangeBars = [(i / j) * 100 for i,j in zip(not_survived__embarked, totals)]
 
# plot
barWidth = 0.85
names = ('Cherbourg','Queenstown','Southampton')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label='Survived')
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label='Not survived')
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Port of embarkation")

plt.legend()
# Show graphic
plt.show()

###############################################################################
# Stacked Chart for SibSp

survival_SibSp = df.groupby(['Survived','SibSp'])['PassengerId'].count()
survival_SibSp

survived_SibSp = survival_SibSp.loc[1]
not_survived_SibSp = survival_SibSp.loc[0]

survival_SibSp = pd.concat([survived_SibSp,not_survived_SibSp], axis=1, keys=['Survived_SibSp','Not_Survived_SibSp'])
survival_SibSp.fillna(0, inplace=True)

survived_SibSp = survival_SibSp['Survived_SibSp']
not_survived_SibSp = survival_SibSp['Not_Survived_SibSp']

# From raw value to percentage
totals = [i+j for i,j in zip(survived_SibSp, not_survived_SibSp)]
greenBars = [(i / j) * 100 for i,j in zip(survived_SibSp, totals)]
orangeBars = [(i / j) * 100 for i,j in zip(not_survived_SibSp, totals)]
 
# plot
barWidth = 0.85

# Create green Bars
plt.bar([0,1,2,3,4,5,8], greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label='Survived')
# Create orange Bars
plt.bar([0,1,2,3,4,5,8], orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label='Not survived')
 
# Custom x axis
plt.xlabel("No. of Siblings/Spouse")

plt.legend()
# Show graphic
plt.show()

###############################################################################
# Stacked Chart for parch

survival_Parch = df.groupby(['Survived','Parch'])['PassengerId'].count()
survival_Parch

survived_Parch = survival_Parch.loc[1]
not_survived_Parch = survival_Parch.loc[0]

survival_Parch = pd.concat([survived_Parch,not_survived_Parch], axis=1, keys=['survived_Parch','not_survived_Parch'])
survival_Parch.fillna(0, inplace=True)

survived_Parch = survival_Parch['survived_Parch']
not_survived_Parch = survival_Parch['not_survived_Parch']

# From raw value to percentage
totals = [i+j for i,j in zip(survived_Parch, not_survived_Parch)]
greenBars = [(i / j) * 100 for i,j in zip(survived_Parch, totals)]
orangeBars = [(i / j) * 100 for i,j in zip(not_survived_Parch, totals)]
 
# plot
barWidth = 0.85

# Create green Bars
plt.bar([0,1,2,3,4,5,6], greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label='Survived')
# Create orange Bars
plt.bar([0,1,2,3,4,5,6], orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label='Not survived')
 
# Custom x axis
plt.xlabel("No. of Siblings/Spouse")

plt.legend()
# Show graphic
plt.show()

###############################################################################
#- Explore Ticket and Fare and tell me if we want to keep them
df.Ticket.head(10)
df.Ticket.tail(10)
#We would drop the ticket

#Survival By Fare
plt.subplot(2,1,1)
sns.distplot(df[df['Survived']==1].Fare.dropna(), 
                  #bins=18, 
                  label = 'Fare for Survived', 
                  kde =False)
plt.xlim(0,200)
plt.legend()

plt.subplot(2,1,2)
sns.distplot(df[df['Survived']==0].Fare.dropna(), 
                  #bins=40, 
                  label = 'Fare for Not survived', 
                  kde =False)
plt.xlim(0,50)
plt.legend()
plt.show()

#We want to keep fare

#Survival By Age
plt.subplot(2,1,1)
sns.distplot(df[df['Survived']==1].Age.dropna(), 
                  #bins=18, 
                  label = 'Fare for Survived', 
                  kde =False, 
                  color = 'red',
                  hist_kws={'alpha':0.1})
plt.legend()
plt.subplot(2,1,2)
sns.distplot(df[df['Survived']==0].Age.dropna(), 
                  #bins=40, 
                  label = 'Fare for Not survived', 
                  kde =False,
                  color = 'blue',
                  hist_kws={'alpha':0.1})
plt.legend()
plt.show()

#Survival By Age part 2

sns.distplot(df[df['Survived']==1].Age.dropna(), 
                  #bins=18, 
                  label = 'Age for Survived', 
                  kde =False, 
                  color = 'red',
                  hist_kws={'alpha':0.1})
sns.distplot(df[df['Survived']==0].Age.dropna(), 
                  #bins=40, 
                  label = 'Age for Not survived', 
                  kde =False,
                  color = 'blue',
                  hist_kws={'alpha':0.1})
plt.legend()
plt.show()

#- Check continuous numerical variables for outliers

plt.boxplot(df.Fare)
plt.boxplot(df.Age.dropna())

###############################################################################
#5. Train a classification algorithm

from sklearn.linear_model import LogisticRegression

"""
#Preprocessing (Make the changes into both the datasets)
- convert non-numeric features into numeric
- features have widely different ranges, need to convert into roughly the same scale
"""

train_file = r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\Titanic dataset\train.csv'
test_file = r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\Titanic dataset\test.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

train.columns
test.columns

#Let's drop the variables we decided from both the dataframes

train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
train.columns

test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test.columns

#Let's treat the missing values
train.isnull().sum()
test.isnull().sum()

#Within, train, we have to treat Age and Embarked and within Test we have to treat Age and Fare

train.Age.fillna(train.Age.median(), inplace=True)

most_frequent = train.groupby('Embarked')['Survived'].count().reset_index().max()['Embarked']
train.Embarked.fillna(most_frequent, inplace=True)

test.Age.fillna(test.Age.median(), inplace=True)
test.Fare.fillna(test.Fare.median(), inplace=True)

train.isnull().sum()
test.isnull().sum()

#Convert the text/factor variables into numeric - Sex and Embarked

#Sex

train.groupby('Sex').groups.keys()
test.groupby('Sex').groups.keys()

sex = {'female': 1, 'male': 0}

train['Sex'] = train.Sex.map(sex)
test['Sex'] = test.Sex.map(sex)

train.groupby('Sex').groups.keys()
test.groupby('Sex').groups.keys()

#Embarked

train.groupby('Embarked').groups.keys()
test.groupby('Embarked').groups.keys()

port = {'S': 0, 'C': 1, 'Q':2}

train.groupby('Embarked')['Survived'].count()

train['Embarked'] = train.Embarked.map(port)
test['Embarked'] = test.Embarked.map(port)

train.groupby('Embarked').groups.keys()
test.groupby('Embarked').groups.keys()

#We will skip creating new features and creating categories or extracting information for now
#We will skip removing the outliers
#We will skip normalizing or standardizing the data
##########################################################
#####################
#Training algorithm 

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
