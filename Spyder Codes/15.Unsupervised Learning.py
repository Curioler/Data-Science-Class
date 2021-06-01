# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:22:53 2020

@author: Bhavin
"""

#Did anybody solve the multiple regression algo problem for house prices?

"""
Unsupervised learning is a machine learning technique, where you do not need 
to supervise the model. Instead, you need to allow the model to work on its 
own to discover information. It mainly deals with the unlabelled data. 

Here, are prime reasons for using Unsupervised Learning:

Unsupervised machine learning finds all kind of unknown patterns in data.


###############################################################################
Clustering: An unsupervised learning technique
Clustering is an important concept when it comes to unsupervised learning. 
It mainly deals with finding a structure or pattern in a collection of 
uncategorized data. Clustering algorithms will process your data and find 
natural clusters(groups) if they exist in the data. You can also modify how 
many clusters your algorithms should identify. It allows you to adjust the 
granularity of these groups. 

###############################################################################

Some applications of unsupervised machine learning techniques are:

- Clustering automatically split the dataset into groups base on their 
    similarities
- Anomaly detection can discover unusual data points in your dataset. It is 
    useful for finding fraudulent transactions
- Association mining identifies sets of items which often occur together in 
    your dataset
- Latent variable models are widely used for data preprocessing. Like reducing 
    the number of features in a dataset or decomposing the dataset into multiple 
    components
    
Some good links to see more resources:
    
- https://www.guru99.com/unsupervised-machine-learning.html
- https://towardsdatascience.com/unsupervised-learning-and-data-clustering-eeecb78b422a
- https://theappsolutions.com/blog/development/unsupervised-machine-learning/ 
"""

#import the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

###############################################################################
#Simple example

from sklearn.cluster import KMeans 

# Check out the documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from scipy.spatial.distance import cdist 

#Check out documentation - https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

#Creating the data 
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8]) 
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3]) 
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2) 

#Visualizing the data 
plt.scatter(x1, x2) 
plt.xlim([0, 10]) 
plt.ylim([0, 10]) 
plt.title('Dataset') 
plt.show() 

#How many clusters should be formed? What is the best grouping scheme?

#Visualiazation might not give you the answer all the time, so, Distortion and Inertia 

"""
Distortion: It is calculated as the average of the squared distances from the 
cluster centers of the respective clusters. Typically, the Euclidean distance metric is used.

Inertia: It is the sum of squared distances of samples to their closest cluster center.
"""

#Building the clustering model and calculating the values of the Distortion and Inertia

distortions = [] 
inertias = [] 
K = range(1,10) 

for k in K: 
	#Building and fitting the model 
	kmeanModel = KMeans(n_clusters=k).fit(X) 
	kmeanModel.fit(X)	 
	
	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
					'euclidean'),axis=1)) / X.shape[0]) 
	inertias.append(kmeanModel.inertia_) 


#Distortion
k_distortion = np.array(list(zip(K, distortions))).reshape(len(K), 2) 
k_distortion

plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 

#Inertia
k_inertia = np.array(list(zip(K, inertias))).reshape(len(K), 2) 
k_inertia

plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show() 

kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(kmeansmodel.cluster_centers_[:,0], kmeansmodel.cluster_centers_[:,1], s=100, c = 'black', label = 'Centroids')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Clustering of X')
plt.legend()
plt.show()

###############################################################################
#Let's try it with 4 clusters

kmeansmodel = KMeans(n_clusters= 4, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], c = 'yellow', label = 'Cluster 4')
plt.scatter(kmeansmodel.cluster_centers_[:,0], kmeansmodel.cluster_centers_[:,1], s=100, c = 'black', label = 'Centroids')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Clustering of X')
plt.legend()
plt.show()

###############################################################################
#And now with 2 clusters

kmeansmodel = KMeans(n_clusters= 2, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(kmeansmodel.cluster_centers_[:,0], kmeansmodel.cluster_centers_[:,1], s=100, c = 'black', label = 'Centroids')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Clustering of X')
plt.legend()
plt.show()

###############################################################################
df = pd.read_csv(r'C:\Users\Bhavin\Google Drive\Data Science Class\Assets\Mall_Customers.csv')

#Data Exploration
df.head()
df.tail()
df.shape
df.info()

"""
Observations:
- 5 columns, ID is to be dropped
- 200 rows
- Gender is to be converted to numeric
- No missing values
- Numeric ranges are different so standard scaler might be required
"""

### Feature sleection for the model
#Considering only 2 features (Annual income and Spending Score) and no Label available
X= df.iloc[:, [3,4]].values

#Building the Model
#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod
#to figure out K for KMeans, use ELBOW Method on KMEANS++ Calculation

from sklearn.cluster import KMeans
inertia=[]

#we always assume the max number of cluster would be 10

###get optimum no of clusters

for i in range(1,21):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

    #inertia_ is the formula used to segregate the data points into clusters

#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1,21), inertia, '-bx')
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('Inertia')
plt.show()

#last elbow comes at k=5
"""
no matter what range we select ex- (1,21) also i will see the same behaviour 
but if we chose higher range it is little difficult to visualize the ELBOW
that is why we usually prefer range (1,11)
Finally we got that k=5
"""

#Model Build
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

"""
This use case is very common and it is used in BFS industry(credit card) and 
retail for customer segmenattion.
"""

#Visualizing all the clusters 

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

"""
Model Interpretation 

#Cluster 1 (Red Color) -> earning high but spending less
#cluster 2 (Blue Colr) -> average in terms of earning and spending 
#cluster 3 (Green Color) -> earning high and also spending high [TARGET SET]
#cluster 4 (cyan Color) -> earning less but spending more
#Cluster 5 (magenta Color) -> Earning less , spending less

We can put Cluster 3 into some alerting system where email can be sent to 
them on daily basis as these are easy to converse wherein others we can set 
like once in a week or once in a month
"""

###############################################################################
"""
What if we are interested in selling some expensive jewelleries to women and 
want to give them offers?
"""
#Let's create the coded variable for gender
df = pd.get_dummies(df, columns=["Gender"])

X = df.iloc[:,2:-1].values

from sklearn.cluster import KMeans
inertia=[]

#get optimum no of clusters

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1,11), inertia, '-bx')
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('Inertia')
plt.show()

#Model Build
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

#Visualizing all the clusters 

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
###############################################################################

"""
Run clustering algorithm on Iris flower dataset and compare it with the actual
result using confusion matrix

Is unsupervised learning better in predicting the group then supervised learning?
"""