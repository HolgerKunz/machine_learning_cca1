
# coding: utf-8

## Module 2: Build a classifier based on KNN

# ## Overview of KNN
# 
# Learning outcomes:
# 
# - Understand what KNN is trying to solve
# 
# - Understand how KNN works and its downsides
# 
# There are many machine learning algorithms available; here we'll go into brief detail on two common and interesting ones: K-nearest neighbour and Naive Bayes.
# 
# ### Introduction to KNN algorithm
# 
# sklearn.neighbors provides functionality for unsupervised and supervised neighbors-based learning methods. Unsupervised nearest neighbors is the foundation of many other learning methods, notably manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels.
# 
# The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree.).
# 
# Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems, including handwritten digits or satellite image scenes. Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.
# 
# The classes in sklearn.neighbors can handle either Numpy arrays or scipy.sparse matrices as input. For dense matrices, a large number of possible distance metrics are supported. For sparse matrices, arbitrary Minkowski metrics are supported for searches.
# 
# ### Example of learning routines that rely on nearest neighbours
# 
# There are many learning routines which rely on nearest neighbors at their core. One example is kernel density estimation, discussed in the density estimation section.
# 
# ### Downsides of KNN
# 
#   - High computational cost
#   
#   - Weighted data points can be too high

# ## (Load Breast cancer dataset)

# In[2]:

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

from sklearn.cross_validation import train_test_split
#Split into training and test data:
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Check:
print 'X_train dimensions: ', X_train.shape
print 'y_train dimensions: ', y_train.shape
print 'X_test dimensions: ', X_test.shape
print 'y_test dimensions: ', y_test.shape


outfile = "/Users/santia01/Copy/Personal_Proj_Ongoing/MyProjects/CCA_python/github/machine_learning_cca1/BCdata"
np.savez(outfile, X, y)

#npz = np.load("BCdata.npz")
#X = npz['arr_0']
#y = npz['arr_1']


# ### Learning activity 1: Find nearest neighbour with breast cancer example data
# 
# Learning outcomes:
# 
# - Understand how to run scikit-learn
# 
# - Understand how to configure KNN in scikit-learn
# 
# - Apply KNN to a real dataset
# 
# ### 1.1. Introduction to classes from sklearn.neighbors library
# 
# NearestNeighbors implements unsupervised nearest neighbors learning. It acts as a uniform interface to three different nearest neighbors algorithms: BallTree, KDTree, and a brute-force algorithm based on routines in sklearn.metrics.pairwise. The choice of neighbors search algorithm is controlled through the keyword 'algorithm', which must be one of ['auto', 'ball_tree', 'kd_tree', 'brute']. When the default value 'auto' is passed, the algorithm attempts to determine the best approach from the training data. 
# 
# ``Warning``: Regarding the Nearest Neighbors algorithms, if two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on the ordering of the training data.
# 
# The Ball Tree and KD Tree have the same interface; we’ll show an example of using the ball_tree here:

# In[3]:

from sklearn import neighbors

nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

distances[:5,]


# In[4]:

indices[:5]


# Because the query set matches the training set, the nearest neighbour of each point is the point itself, at a distance of zero.
# It is also possible to efficiently produce a sparse graph showing the connections between neighboring points:

# In[5]:

import numpy as np
BCgraph = nbrs.kneighbors_graph(X).toarray()
print BCgraph
print sum(BCgraph[0]) #5 neighbors
print sum(BCgraph[1])


# Our dataset is structured such that points nearby in index order are nearby in parameter space, leading to an approximately block-diagonal matrix of K-nearest neighbors.

# In[65]:

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt

#Plot heatmap of sparce 
fig, ax = plt.subplots()
heatmap = ax.pcolor(BCgraph, cmap=plt.cm.Blues)
plt.show()


# #### Exercise: Explore options available for neighbour search

# In[66]:

#n_neighbors

nbrs = neighbors.NearestNeighbors(n_neighbors=3)
nbrs = nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)

print distances[:5,]
print indices[:5]

nbrs = neighbors.NearestNeighbors(n_neighbors=10).fit(X)
distances, indices = nbrs.kneighbors(X)

print distances[:5,]
print indices[:5]


# ##Learning activity 2:  Apply nearest neighbours algorithm with scikit-learn [20 min] [20 min]
# 
# Learning outcomes:
# 
# - Understand how KNN classification works  
# 
# - Classify using scikit-learn implemented classifiers
# 
# - Understand differences between classifiers
# 
# ### 2.1. Introduction to classifiers and how they work
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

# ### 2.2. Calssification

# In[68]:

#Classification based on 3 nearest neighbours
knnK3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knnK3 = knnK3.fit(X, y)
predictedK = knnK3.predict(X)

#Classification based on 15 nearest neighbours 
knnK15 = neighbors.KNeighborsClassifier(n_neighbors=15)
knnK15= knnK15.fit(X, y)
predictedK15 = knnK15.predict(X)


# #### Weights
# 
# The basic nearest neighbors classification uses uniform weights: that is, the value assigned to a query point is computed from a simple majority vote of the nearest neighbors. Under some circumstances, it is better to weight the neighbors such that nearer neighbors contribute more to the fit. This can be accomplished through the weights keyword. The default value, weights = 'uniform', assigns uniform weights to each neighbor. weights = 'distance' assigns weights proportional to the inverse of the distance from the query point. Alternatively, a user-defined function of the distance can be supplied which is used to compute the weights.

# In[57]:

#Classification based on 3 nearest neighbours
knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
knn = knn.fit(X, y)


### Learning activity 3: Calculate the accuracy and recall scores of your classifier [20min]

# Learning outcomes:
# 
# - Know the metrics by which models can be evaluated
# 
# - Understand how to evaluate performance in terms of accuracy and recall
# 
# - Know how to use cross validation
# 
# ### 3.1. Describe metrics to validate classifier
# 
# Evaluation functions allow you to distinguish if the learner has made good predictions by evaluating the classifier (e.g., whether the spam filter correctly classifies previously unseen emails as spam or not spam).
# When experimenting with learning algorithms, it is important not to test the prediction of an estimator on the data used to fit the estimator.
# (Describe: precision/accuracy and recall)
# 
# ### 3.2. Compare number of correct predictions (loop) - Demonstrate classification result

# In[3]:

from sklearn.cross_validation import train_test_split

#Split into training and test data:
XTrain, XTest, yTrain, yTest = train_test_split(X, y)

#Check:
print 'XTrain dimensions: ', XTrain.shape
print 'yTrain dimensions: ', yTrain.shape
print 'XTest dimensions: ', XTest.shape
print 'yTest dimensions: ', yTest.shape


# In[23]:

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(XTrain, yTrain)

predicted = knn.predict(XTest)

from sklearn import metrics
mat = metrics.confusion_matrix(yTest, predicted)
mat 


# In[24]:

a = np.array(map(str, zip(yTest,predicted)))

tp = sum(a == '(0, 0)')
tn = sum(a == '(1, 1)')
fp = sum(a == '(1, 0)')
fn = sum(a == '(0, 1)')
print "true positives", tp
print "true negatives", tn
print "false positives", fp
print "false negatives", fn

print "correct calls:", sum(yTest == predicted)
print "correct calls:", tp+tn
print "number of classifications:", yTest.size

#Accuracy is the total of correct calls / total of classifications
accuracy = sum(yTest == predicted) / float(yTest.size)

#Precision is the number of true positives / total number of elements the 'predicted' class
precisionM = mat[0,0] / float(mat[0,0]+mat[1,0])
precisionB = mat[1,1] / float(mat[0,1]+mat[1,1])

#Recall is the number of true positives / total number of elements the 'known' class
recallM = mat[0,0] / float(mat[0,0]+mat[0,1])
recallB = mat[1,1] / float(mat[1,0]+mat[1,1])

#print out
print "accuracy:", accuracy
print "precision for class M:", round(precisionM,2)
print "precision for class B:", round(precisionB,2)
print "recall for class M:", round(recallM,2)
print "recall for class B:", round(recallB,2)


# In[25]:

from sklearn import metrics
print metrics.classification_report(yTest, predicted)
print 'accuracy:', knn.score(XTest, yTest)
#print metrics.f1_score(yTest, predicted)


# #### Use randomised data
# 
# Compare with model trained on randomised X data:

# In[26]:

import numpy 

perm = np.random.permutation(y_train.size)
X_random = numpy.copy(X_train)[perm]

knn2 = neighbors.KNeighborsClassifier()

#Compare with model trained on randomised X data:
knn2 = knn2.fit(X_random, y_train) 
predicted = knn2.predict(X_random)
print float(sum(y_train != predicted))/len(y_train)
print 'Proportion mislabeled random model: %.5f'  %(1- knn2.score(X_random, y_train))


# ## Learning activity 4:  Demonstrate the classification result by plotting accuracy for different options

# In[28]:

get_ipython().magic(u'matplotlib inline')


#use different distance metrics
#http://blog.yhathq.com/posts/classification-using-knn-and-python.html

import matplotlib.pyplot as plt
from sklearn import neighbors

def plotvector(XTrain, yTrain, XTest, yTest, weights):
    results = []
    
    #Xtrain = XTrain[:, :2]
    
    for n in range(1, 310, 2):
        clf = neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights)
        clf = clf.fit(XTrain, yTrain)
        preds = clf.predict(XTest)
        accuracy = clf.score(XTest, yTest)
        results.append([n, accuracy])
 
    results = np.array(results)
    return(results)

pltvector1 = plotvector(XTrain, yTrain, XTest, yTest, weights="uniform")
pltvector2 = plotvector(XTrain, yTrain, XTest, yTest, weights="distance")
#pltvector3 = plotvector(XTrain, yTrain, XTest, yTest, weights="uniform",algorithm="kd_tree")
#pltvector4 = plotvector(XTrain, yTrain, XTest, yTest, weights="distance",algorithm="kd_tree")
line1 = plt.plot(pltvector1[:,0], pltvector1[:,1], label="uniform")
line2 = plt.plot(pltvector2[:,0], pltvector2[:,1],  label="distance")
#line3 = plt.plot(pltvector3[:,0], pltvector3[:,1], label="uniform_KDtree")
#line4 = plt.plot(pltvector4[:,0], pltvector4[:,1],  label="distance_KDtree")
plt.legend(loc=3)
plt.ylim(0.5, 1)
plt.title("Accuracy with Increasing K")
plt.show()

zip(pltvector1[:10],pltvector2[:10])


# ### Demonstrate the classification result by ploting the decision boundery
# 
# Sample usage of Nearest Neighbors classification. It will plot the decision boundaries for each class:

# In[17]:

#code taken from: http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py 
    
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def decision_plot(XTrain, yTrain, n_neighbors, weights):
    h = .02  # step size in the mesh
    
    Xtrain = XTrain[:, :2] # we only take the first two features.
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(Xtrain, yTrain)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=yTrain, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.show()


# In[18]:

get_ipython().magic(u'matplotlib inline')

decision_plot(XTrain, yTrain, n_neighbors=45, weights='uniform')
decision_plot(XTrain, yTrain, n_neighbors=45, weights='distance')
decision_plot(XTrain, yTrain, n_neighbors=3, weights='uniform')
decision_plot(XTrain, yTrain, n_neighbors=3, weights='distance')


# ## Learning activity 5: Implement k-fold cross-validation for your model

# In[33]:

print XTrain.shape
print XTest.shape


# In[34]:

from sklearn import cross_validation 

#Cross validation score
from sklearn import cross_validation
scores = cross_validation.cross_val_score(knn, XTrain, yTrain, cv=5)
print "split into 5:", scores

#Cross validation score (leave one out)
#from sklearn.cross_validation import cross_val_score
#scores = cross_val_score(knn, XTrain, yTrain, cv=yTrain.size) #the same as leave one out
#print "Leave one out:", scores



# In[35]:

scores


# In[36]:

scores.mean()


# # Further reading
# 
# - http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms - For a discussion of the strengths and weaknesses of each option, see Nearest Neighbor Algorithms.
# - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html - KNeighborsClassifier documentation

# In[ ]:



