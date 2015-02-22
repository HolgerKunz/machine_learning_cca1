
#Numoy

from numpy import random

# Generating a random array
X = np.random.random((10, 5)) # a 10 x 5 array
X

# Accessing elements
# get a single element
print "single element: %f" %X[0, 0]
# get a row
X[1]
print "row: %s" %' '.join('%10.6f'%F for F in X[1] )
# get a column
X[:, 1]
print "Column: %s" %' '.join('%10.6f'%F for F in X[:, 1])


#Scikit-learn data
from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()
n_samples, n_features = iris.data.shape
n_samples, n_features
iris.data[:10]

#Assessin elements within iris dataset
iris.target[[50,20]]
iris.target
iris.target[45:55]
list(iris.target_names)[iris.target[45:55]]

iris.data.shape
iris.target.shape

#Two dimenstions visualisation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

X = iris.data  # we only take the first two features.
print X[:10,]
Y = iris.target
y

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# Plot the training points
plt.scatter(X[:, 2], X[:, 3], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Petal width')

#to do
#Pairs of correlation plot
#PCA to show how supervised Vs unsupervised (??)  [ See how much time this would take ]
#Classification (quick...) to show result.. (??) [ See how much time this would take ]
