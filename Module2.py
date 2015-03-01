
# coding: utf-8

## Module 2: Build a classifier based on KNN

# ## Learning activity 1: Overview of KNN [20 min]
# 
# Learning outcomes:
# 
# - Understand what KNN is trying to solve
# 
# - Understand how KNN works and its downsides
# 
# There are many machine learning algorithms available; here we'll go into brief detail on two common and interesting ones: K-nearest neighbour and Naive Bayes.
# 
# ### 1.1. Introduction to KNN algorithm
# 
# sklearn.neighbors provides functionality for unsupervised and supervised neighbors-based learning methods. Unsupervised nearest neighbors is the foundation of many other learning methods, notably manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels.
# 
# The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree.).
# 
# Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems, including handwritten digits or satellite image scenes. Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.
# 
# The classes in sklearn.neighbors can handle either Numpy arrays or scipy.sparse matrices as input. For dense matrices, a large number of possible distance metrics are supported. For sparse matrices, arbitrary Minkowski metrics are supported for searches.
# 
# ### 1.2. Example of learning routines that rely on nearest neighbours
# 
# There are many learning routines which rely on nearest neighbors at their core. One example is kernel density estimation, discussed in the density estimation section.
# 
# ### 1.3. Downsides of KNN
# 
#   - High computational cost
#   
#   - Weighted data points can be too high

# ## (Load Breast cancer dataset)

# In[114]:

import csv
import scipy
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

filename = "breast-cancer-wisconsin/wdbc.data"

#Labels for the outputs
output_labels = ["M", "B"]
data = list(csv.reader(open(filename, 'rU' )))
data = scipy.array(data)

#Get predictors (X): columns 0, and 2:31
X = data[0:,2:31]
X = X.astype(np.float) #Convert to array of floats 
print X

#Get targets (y): column 1
y = data[0:, 1]
print y[:10]

#Convert the labels into numerical values:
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y) #Fit numerical values to the categorical data.
y = label_encoder.transform(y) #Transform the labels into numerical correspondents.

#Check label conversion:
print y[:10]
print 'Converted: ', y[0]

#Check all data has been successfully loaded (array shape):
print 'X dimensions: ', X.shape
print 'y dimensions: ', y.shape

#Split into training and test data:
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Check:
print 'X_train dimensions: ', X_train.shape
print 'y_train dimensions: ', y_train.shape
print 'X_test dimensions: ', X_test.shape
print 'y_test dimensions: ', y_test.shape


# ## Learning activity 2: Apply nearest neighbours algorithm with scikit-learn [20 min]
# 
# Learning outcomes:
# 
# - Understand how to run scikit-learn
# 
# - Understand how to configure KNN in scikit-learn
# 
# - Apply KNN to a real dataset
# 
# ### 2.1. Introduction to classes from sklearn.neighbors library
# 
# NearestNeighbors implements unsupervised nearest neighbors learning. It acts as a uniform interface to three different nearest neighbors algorithms: BallTree, KDTree, and a brute-force algorithm based on routines in sklearn.metrics.pairwise. The choice of neighbors search algorithm is controlled through the keyword 'algorithm', which must be one of ['auto', 'ball_tree', 'kd_tree', 'brute']. When the default value 'auto' is passed, the algorithm attempts to determine the best approach from the training data. 
# 
# ``Warning``: Regarding the Nearest Neighbors algorithms, if two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on the ordering of the training data.

# ### 2.2. Find nearest neighbour with breast cancer example data
# 
# The Ball Tree and KD Tree have the same interface; we’ll show an example of using the ball_tree here:

# In[33]:

from sklearn import neighbors

nbrs = neighbors.NearestNeighbors(algorithm='ball_tree').fit(X_train)
distances, indices = nbrs.kneighbors(X_train)

print distances[:10,]
print indices[:10]


# Because the query set matches the training set, the nearest neighbour of each point is the point itself, at a distance of zero.
# It is also possible to efficiently produce a sparse graph showing the connections between neighboring points:

# In[41]:

import numpy as np
BCgraph = nbrs.kneighbors_graph(X_train).toarray()
print BCgraph
print sum(BCgraph[0]) #5 neighbors
print sum(BCgraph[1])


# Our dataset is structured such that points nearby in index order are nearby in parameter space, leading to an approximately block-diagonal matrix of K-nearest neighbors.

# In[35]:

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt

#Plot heatmap of sparce 
fig, ax = plt.subplots()
heatmap = ax.pcolor(BCgraph, cmap=plt.cm.Blues)
plt.show()


# ### 2.3. Explore options available for neighbour search

# In[45]:

#n_neighbors

nbrs = neighbors.NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(X_train)
distances, indices = nbrs.kneighbors(X_train)

print distances[:5,]
print indices[:5]

nbrs = neighbors.NearestNeighbors(n_neighbors=3,algorithm='ball_tree').fit(X_train)
distances, indices = nbrs.kneighbors(X_train)

print distances[:5,]
print indices[:5]


# ##Learning activity 3: Classification [20 min]
# 
# Learning outcomes:
# 
# - Understand how KNN classification works  
# 
# - Classify using scikit-learn implemented classifiers
# 
# - Understand differences between classifiers
# 
# ### 3.1. Introduction to classifiers and how they work
# 
# Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.
# 
# scikit-learn implements two different nearest neighbors classifiers: KNeighborsClassifier implements learning based on the k nearest neighbors of each query point, where k is an integer value specified by the user. RadiusNeighborsClassifier implements learning based on the number of neighbors within a fixed radius r of each training point, where r is a floating-point value specified by the user.
# 
# #### KNeighborsClassifier
# The k-neighbors classification in KNeighborsClassifier is the more commonly used of the two techniques. The optimal choice of the value k is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct.
# 
# #### RadiusNeighborsClassifier
# In cases where the data is not uniformly sampled, radius-based neighbors classification in RadiusNeighborsClassifier can be a better choice. The user specifies a fixed radius r, such that points in sparser neighborhoods use fewer nearest neighbors for the classification. For high-dimensional parameter spaces, this method becomes less effective due to the so-called “curse of dimensionality”.
# 
# #### Weights
# 
# The basic nearest neighbors classification uses uniform weights: that is, the value assigned to a query point is computed from a simple majority vote of the nearest neighbors. Under some circumstances, it is better to weight the neighbors such that nearer neighbors contribute more to the fit. This can be accomplished through the weights keyword. The default value, weights = 'uniform', assigns uniform weights to each neighbor. weights = 'distance' assigns weights proportional to the inverse of the distance from the query point. Alternatively, a user-defined function of the distance can be supplied which is used to compute the weights.

# ### 3.2. Calssification

# In[124]:

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='ball_tree')
knn.fit(X_train, y_train) 
predicted = knn.predict(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree')
knn.fit(X_train, y_train) 
predicted = knn.predict(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='ball_tree')
knn.fit(X_train, y_train) 
predicted = knn.predict(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='ball_tree')
knn.fit(X_train, y_train) 
predicted = knn.predict(X_test)


# ### 3.3. Demonstrate the classification result by ploting the decision boundery
# 
# Sample usage of Nearest Neighbors classification. It will plot the decision boundaries for each class:

# In[125]:

import matplotlib.pyplot as plt

def decision_plot(X_train, y_train, n_neighbors, weights):
    h = .02  # step size in the mesh
    
    Xtest = X_train[:, :2] # we only take the first two features.
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm='ball_tree')
    clf.fit(Xtest, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtest[:, 0].min() - 1, Xtest[:, 0].max() + 1
    y_min, y_max = Xtest[:, 1].min() - 1, Xtest[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=y_train, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.show()


# In[126]:

decision_plot(X_train, y_train, n_neighbors=15, weights='uniform')
decision_plot(X_train, y_train, n_neighbors=15, weights='uniform')
decision_plot(X_train, y_train, n_neighbors=5, weights='distance')
decision_plot(X_train, y_train, n_neighbors=5, weights='distance')


### Learning activity 4: Validation metrics [20min]

# Learning outcomes:
# 
# - Know the metrics by which models can be evaluated
# 
# - Understand how to evaluate performance in terms of accuracy and recall
# 
# - Know how to use cross validation
# 
# ### 4.1. Describe metrics to validate classifier
# 
# When experimenting with learning algorithms, it is important not to test the prediction of an estimator on the data used to fit the estimator.
# (Describe: precision/accuracy and recall)
# 
# ### 4.2. Compare number of correct predictions

# In[139]:

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='ball_tree')
knn = knn.fit(X_train, y_train) 

# Validate classifier
# y_pred = knn.predict(X_train)
# print 'Proportion mislabeled: %.5f' %(float(sum(y_train != y_pred))/len(y_train))
print 'Proportion mislabeled train set: %.5f'  %(1 - knn.score(X_train, y_train))

# Now see how performs with the test data
# y_pred_test = knn.predict(X_test)
# print 'Proportion mislabeled: %.5f' %(float(sum(y_test != y_pred_test))/len(y_test))
print 'Proportion mislabeled test set: %.5f'  %(1 - knn.score(X_test, y_test))


# #### Use randomised data
# 
# Compare with model trained on randomised X data:

# In[143]:

import numpy 

perm = np.random.permutation(y_train.size)
X_random = numpy.copy(X_train)[perm]

knn = neighbors.KNeighborsClassifier()

#Compare with model trained on randomised X data:
knn = knn.fit(X_random, y_train) 
predicted = knn.predict(X_random)
print float(sum(y_train != predicted))/len(y_train)
print 'Proportion mislabeled random model: %.5f'  %(1- knn.score(X_random, y_train))


# ### 4.2. Cross validation

# In[147]:


from sklearn import cross_validation 

knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='ball_tree')
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)

from sklearn import metrics
print metrics.classification_report(y_test, predicted)
print metrics.confusion_matrix(y_test, predicted)
#print metrics.f1_score(y_test, predicted)

#Cross validation score
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(knn, X_train, y_train, cv=5) #cv is the number of samples the data is split into
print "split into 5:", scores

#Cross validation score (leave one out)
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(knn, X_train, y_train, cv=y_train.size) #the same as leave one out
print "Leave one out:", scores


# # Further reading
# 
# - http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms - For a discussion of the strengths and weaknesses of each option, see Nearest Neighbor Algorithms.
