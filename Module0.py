
# coding: utf-8

## Module 0: Overview of machine learning practice

### Goals of this tutorial

# - Introduce the basics of Machine Learning, and some skills useful in practice.
# - Introduce the syntax of scikit-learn, so that you can make use of the rich toolset available.

# ## Learning activity 1: Introduction to Machine Learning [5 min]
# 
# Learning outcomes:
# 
# - Understand machine learning basics
# 
# - Know what type of tasks are good for machine learning
# 
# - Understand machine learning methodology through some simple examples
# 
# 
# ### 1.1. What is machine learning
# 
# Machine learning is a field of computer science that deals with the construction 
# of algorithms that can learn from data and can be used to make predictions or decisions. 
# Machine learning algorithms provide computers with the ability to search through data, 
# look for patterns, and uses extracted information to make educated decisions without the 
# computer being explicitly programmed with static instructions.
# 
# ### 1.2. Machine learning examples
# 
# Lets run through some examples of machine learning problems.
# 
# #### Recommender systems
# 
# Netflix is a movie rental company in the US. Netflix has a movie recommender system that 
# changes according to the user's personal movie reantals. If a user frequently gives high 
# ratings to comedy movies or TV shows, Netflix may want to recommend more comedy-related 
# videos. So, one of the goals of recommender systems is to try and predict what the customers 
# would think of the other movies based on what they rated so far. 

# In[1]:

from IPython.display import display, Image
display(Image(filename='images/movie_recommender.jpg'))


# #### Spam filter
# 
# Another classical example is the email spam detection. In this case we want 
# to build a classifier that will classify an email as "spam" or "not spam". 
# The data in the word clouds below were taken from over thousands of "spam" or "good" emails. 
# The goal of the classifier is to classify spam from email based on the frequency of words in the email. 
# This is a typical text classification example 

# In[2]:

from IPython.display import display, Image
display(Image(filename='images/spam_wordcloud.png'))


# #### Recognizing hand-written digits
# 
# This is another tipical example of machine learning classification procedure, 
# it was one of the first learning tasks that was used to develop neural networks. 
# Consider some handwritten digits below. The goal is to, based on an image of any 
# of these digits, say what the digit is, to classify into the 10 digit classes.
# Well to humans, this looks like a pretty easy task. We're pretty good at pattern recognition. 
# It turns out it's a really difficult task to crack for computers. They're getting better and better all the time.

# In[11]:

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
digits = load_digits()

digits.keys()
n_samples, n_features = digits.data.shape
print (n_samples, n_features)
print digits.data[0]
print digits.target
print digits.data.shape
print digits.images.shape
print np.all(digits.images.reshape((1797, 64)) == digits.data)
print digits.data.__array_interface__['data']
print digits.images.__array_interface__['data']


# In[3]:

from IPython.display import display, Image
display(Image(filename='images/digits_figure.png'))


# Each feature is a real-valued quantity representing the
# darkness of a pixel in an 8x8 image of a hand-written digit.
# 
# Even though each sample has data that is inherently two-dimensional, the data matrix flattens
# this 2D data into a **single vector**, which can be contained in one **row** of the data matrix.

# ## Learning activity 2: Understand difference between supervised and unsupervised problems [10 min]
# 
# Learning outcomes:
# 
# - Understand difference between supervised and unsupervised problems
# 
# - Be able to describe how different algorithms work
# 
# - See how different algorithms can be applied to different problems
# 
# ### 2.1. Supervised and unsupervised problems
# 
# A classifier is a system that inputs a vector of discrete and/or continuous predictor 
# values and outputs a single discrete value, the class. The output variable goes by various names: 
# Y, dependent variable, response or target. The predictor measurments are also called inputs, 
# regressors, covariates, features, or independent variables.
# 
# Machine learning tasks are typically classified into three broad categories, 
# depending on the nature of the learning "signal" or "feedback" available to a learning system. These are:
# 
# - Supervised learning (classification). The data is presented to the computer with example inputs and their desired outputs (labels). Samples belong to discrete lasses and we want to learn a general rule from labeled data how to predict the class of unlabeled data. Finally, evaluation functions allow you to distinguish if the learner has made good predictions by evaluating the classifie (e.g., whether the spam filter correctly classifies previously unseen emails as spam or not spam). 
# 
# - Unsupervised learning. No labels are given to the learning algorithm. The goal is to discover patterns in the data (clustering) or to determine the distribution of data within the input space (density estimation). Dimensionality reduction is also considered a unsupervised learning algorithm. 

# This is a flow chart below was created by scikit-learn contributor [https://github.com/amueller] Andreas Mueller. The flow chart gives a nice summary of 
# which algorithms to choose in various situations!

# In[4]:

#Show scikit-learn cheat sheet: (http://peekaboo-vision.blogspot.co.uk/2013/01/machine-learning-cheat-sheet-for-scikit.html)
from IPython.display import display, Image
display(Image(filename='images/drop_shadows_background.png'))


# Unsupervised learning is an important preprocessor for supervised learning. 
# It's often useful to try to organize your features, choose features based on the X's themselves, 
# and then use those processed or chosen features as input into supervised learning. 
# It is also a lot more common to collect data which is unlabeled. 
# 
# ### 2.2. Trainning vs discovery dataset
# Machine learning is about learning some properties of a data set and applying them to new data. 
# This is why a common practice in machine learning to evaluate an algorithm 
# is to split the data at hand into two sets, one that we call the training set 
# on which we learn data properties and one that we call the testing set on which we test these properties.

# ### Quiz: Is this a suppervised or an unsupervised problem?
# 
# In both cases you are given data, but in one case you have labled data and in the other you have unlabled data.
# 
# - 1) Face recognition
# - 2) Website for comparison-shopping agents
# - 3) Speach reconition
# - 4) Classifying tissue samples
# - 5) Search engine
# - 6) Selecting features from Google's Street View photographic database (trees, pedestrians, cars, traffic lights, lane markers, etc)
# 
# 

# ## Learning activity 3: Scikit learn interface [30 min] 
# 
# Learning outcomes:
# 
# - Understand basics of data manipulation with numpy arrays
# 
# - Understand numpy syntax
# 
# - Know how to create and manipolate numpy arrays
# 
# 
# ### 3.1. Data manipulation with Numpy with simple example exercises
# 
# Being able to manipulate numpy arrays is an important part of doing machine learning in Python. Numpy provides high-performance vector, matrix and higher-dimensional data structures for calculations in Python. 
# 
# In Numpy the terminology used for data sets is array. Numpy is implemented in C and Fortran, which makes computations with numpy arrays memory efficient
# 
# To use numpy need to import the module:

# In[14]:

import numpy as np


# ### 3.2. Creating numpy arrays
# 
# There are a number of ways to create numpy arrays. For example

# In[15]:

# Generate a vector from a Python list 
X = np.array([1,2,3,4])
print X


# In[16]:

# Generate a random array
R = np.random.random((10, 5))  # a 10 x 5 matrix
print R


# In[17]:

# Generate a matrix from a Python nested list
M = np.array([[1, 2], [3, 4]])
print M


# ### 3.3. Properties of numpy arrays
# 
# The shape and the size methods can be applied to numpy ndarray objects:

# In[18]:

#We can get information on the shape of the generated arrays
print "Shape:"
print X.shape
print R.shape
print M.shape

#size gives us the number of elements in the array
print "Size:"
print X.size
print R.size
print M.size

#we can also use the functions np.shape and np.size
print "np.shape and np.size:"
print np.shape(M)
print np.size(M)

print M.itemsize
print M.ndim


# ### 3.4. More ways to create numpy arrays
# 
# We can generate longer arrays automatically using functions, instead of entering the data manually with Python lists. Some of the more common are:

# In[19]:

#More ways to create numpy arrays

#Create a range
x = np.arange(0, 10, 1)
print x

x = np.arange(-1, 1, 0.1)
print x

# using linspace and logspace
print np.linspace(0, 100, 11)
print np.logspace(0, 100, 11, base=10)


# ### 3.5. Manipulating arrays
# 
# We can index elements in an array using the square bracket and indices:

# In[20]:

# X is a 3 x 5 matrix
X = np.random.random((3,5)) 
print X

# get a single element
print X[0, 0]

# get a row
print X[1]
print "row: %s" %' '.join('%10.6f'%F for F in X[1] )

# get a column
print X[:, 1]
print "Column: %s" %' '.join('%10.6f'%F for F in X[:, 1])

# V is a vector with 1 dimension
v = np.array([1,2,3,4])
print v[1]
print "single element of a vector : %d" %v[1]
#print v[1,:] #this does not work!
#print v[:,1] #this does not work!


# ### 3.6. Index slicing
# 
# Index slicing is the technical name for the syntax M[lower:upper:step] to extract part of an array:

# In[21]:

A = np.array([1,2,3,4,5])

# elements from 1 to 3
print A[1:3]

# first three elements 
print A[:3] 

# elements from index 3
print A[3:] 


# Negative indices counts from the end of the array (positive index from the begining):

# In[22]:

# last element
print A[-1] 

# the last three elements
print A[-3:] 


# Slicing applies the same way to multidimensional arrays

# In[23]:

A = np.array([[n+m*10 for n in range(5)] for m in range(5)])
print A

# get a block from the original array
print A[1:4, 1:4]


# ### 3.7. Assign values to arrays
# 
# We can assign new values to elements in an array using indexing:

# In[24]:

M = np.array([[1, 2], [3, 4]])
print M


# In[25]:

M[0,0] = 1100
print M


# In[26]:

# also works for rows and columns
M[1,:] = 0
print M


# In[27]:

#also works for 1D arrays
v = np.array([1,2,3,4])
v[0] = 30
print v


# In[28]:

#Create a sparce matrix (an array with lots of zeros)
#Replace values smaller than 0.7
X = np.random.random((10,5))
X[X < 0.7] = 0
print X


# We get an error if we try to assign the wrong data type to an array

# In[29]:

#Using the dtype we can see what type the data of an array has
X.dtype

#Try to assign a string to the wrong type
M[0,0] = "hello"


# If we want, we can explicitly define the type of the array data when we create it, using the dtype keyword argument. Common type that can be used with dtype are: int, float, complex, bool, object. We can also explicitly define the bit size of the data types, for example: int64, int16, float128, complex128.

# In[30]:

M = np.array([[1, 2], [3, 14]], dtype=complex)
print M


# In[31]:

# Transposing an array
X = np.random.random((3,5))
print X
print X.T


# In[32]:

# Turning a row vector into a column vector
y = np.linspace(0, 100, 5)

print y

# make into a column vector
print y[:, np.newaxis]


# ## Learning Activity 4: Representation of Data in Scikit-learn
# 
# Learning outcomes:
# 
# - Understand basics of data manipulation with numpy arrays
# 
# - Understand numpy syntax
# 
# - Know how to load data into scikit-learn
# 
# - Understand data format and shape within scikit-learn
# 
# ### 4.1. Introduction to Scikit-learn
# 
# When using [scikit-learn](http://scikit-learn.org), it is important to have a handle on how data are represented.
# 
# Machine learning is about creating models from data: for that reason, we'll start by discussing how data can be represented in order to be understood by the computer.  Along with this, we'll build on our matplotlib examples from the previous section and show some examples of how to visualize data.
# 
# Most machine learning algorithms implemented in scikit-learn expect data to be stored in a
# **two-dimensional array or matrix**.  The arrays can be
# either ``numpy`` arrays, or in some cases ``scipy.sparse`` matrices.
# The size of the array is expected to be `[n_samples, n_features]`
# 
# - **n_samples:**   The number of samples: each sample is an item to process (e.g. classify).
#   A sample can be a document, a picture, a sound, a video, an astronomical object,
#   a row in database or CSV file,
#   or whatever you can describe with a fixed set of quantitative traits.
# - **n_features:**  The number of features or distinct traits that can be used to describe each
#   item in a quantitative manner.  Features are generally real-valued, but may be boolean or
#   discrete-valued in some cases.
# 
# The number of features must be fixed in advance. However it can be very high dimensional
# (e.g. millions of features) with most of them being zeros for a given sample. This is a case
# where `scipy.sparse` matrices can be useful, in that they are
# much more memory-efficient than numpy arrays.
# 
# Data in scikit-learn is represented as a **feature matrix** and a **label vector**
# 
# $$
# {\rm feature~matrix:~~~} {\bf X}~=~\left[
# \begin{matrix}
# x_{11} & x_{12} & \cdots & x_{1D}\\
# x_{21} & x_{22} & \cdots & x_{2D}\\
# x_{31} & x_{32} & \cdots & x_{3D}\\
# \vdots & \vdots & \ddots & \vdots\\
# \vdots & \vdots & \ddots & \vdots\\
# x_{N1} & x_{N2} & \cdots & x_{ND}\\
# \end{matrix}
# \right]
# $$
# 
# $$
# {\rm label~vector:~~~} {\bf y}~=~ [y_1, y_2, y_3, \cdots y_N]
# $$
# 
# Here there are $N$ samples and $D$ features.

# ### 4.2. Load an example dataset using scikit-learn: The iris dataset
# 
# scikit.learn has become one of the most popular machine learning packages in Python. It often calls external C-code making the execuation considerably quick. It also with a few standard datasets, 
# we will use the iris dataset in this example. 
# The iris dataset is one of most well-known toy datasets for machine learning.
# 
# There are three species of iris in the dataset (see figure below). 

# In[33]:

from IPython.core.display import Image, display
display(Image(filename='images/iris_setosa.jpg'))
print "Iris Setosa\n"

display(Image(filename='images/iris_versicolor.jpg'))
print "Iris Versicolor\n"

display(Image(filename='images/iris_virginica.jpg'))
print "Iris Virginica"


# ### Quiz Question:
# 
# **If we want to design an algorithm to recognize iris species, what might the features be? What might the labels be?**
# 
# We will need a matrix of `[samples x features]`, and a vector `samples`.
# 
# - What would the `samples` refer to?
# 
# - What might the `features` refer to?
# 
# - What are the X (predictor) and Y (target/response) variables? 

# ### 4.3. Loading the Iris Data with Scikit-Learn
# 
# Scikit-learn has a very straightforward set of data on these iris species.  The data consist of
# the following:
# 
# - Features in the Iris dataset:
# 
#   1. sepal length in cm
#   2. sepal width in cm
#   3. petal length in cm
#   4. petal width in cm
# 
# - Target classes to predict:
# 
#   1. Iris Setosa
#   2. Iris Versicolour
#   3. Iris Virginica
#   
# ``scikit-learn`` embeds a copy of the iris CSV file along with a helper function to load it into numpy arrays:

# In[2]:

from sklearn.datasets import load_iris
iris = load_iris()


# The result is the ``iris`` object, which is basically an dictionary-like object which contains the data.
# Few interesting attributes are: 

# In[3]:

iris.keys()


# - ‘data’, a (samples x features) array with the data to learn 
# - ‘target’, the classification labels
# - ‘target_names’, the meaning of the labels
# - ‘feature_names’, the meaning of the features
# - ‘DESCR’, the full description of the dataset.

# ``iris.data`` gives you the features that can be used to classify the iris sample:

# In[4]:

#subset of first 10 rows:
print iris.data[:10,]


# This iris data is stored in the ``.data member``, which is a n_samples, n_features array. Each row contains one flower instance with information on it's petal and sepal measurments, stored in a 150x4 ``numpy.ndarray`` 

# In[5]:

#samples x features
n_samples, n_features = iris.data.shape
print (n_samples, n_features)


# The feature names correspond to the petal and septal measurments and are given by ``.feature_names``:

# In[6]:

#meaning of the features
print iris.feature_names


# Each flower is labled with 3 different types of irises’ (Setosa, Versicolour, and Virginica). The labels are stored in the ``.target`` member or the iris dataset: 

# In[7]:

#target (numeric)
print iris.target

#shape of arrays
print iris.target.shape


# The names corresponding to each target feature is given by ``.target_names``. 

# In[8]:

#meaning of the target labels 
print iris.target_names
print iris.target_names[iris.target[:10]] #an array was used in-place of an index


# In[9]:

#Full description of the dataset
print iris.DESCR


# ### 4.4. Ploting the iris dataset

# In[35]:

import matplotlib.pyplot as plt

X = iris.data  # we only take the first two features.
print X[:10,]
Y = iris.target
y


# This data is four dimensional, but we can visualize two of the dimensions at a time using a simple scatter-plot:

# In[75]:

get_ipython().magic(u'matplotlib inline')

# Plot the data points
#plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#plt.legend()

for c, i, iris.target_name in zip(["red","green","blue"], [0,1,2], iris.target_names):
    plt.scatter(X[iris.target == i, 0], X[iris.target == i, 1], c=c, label=iris.target_name)
plt.legend(loc='upper right')
plt.show()


# In[73]:

get_ipython().magic(u'matplotlib inline')

# Plot the data points
#plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
#plt.xlabel('Petal length')
#plt.ylabel('Petal width')
#plt.show()

for c, i, iris.target_name in zip(["red","green","blue"], [0,1,2], iris.target_names):
    plt.scatter(X[iris.target == i, 2], X[iris.target == i, 3], c=c, label=iris.target_name)
plt.legend(loc='upper left')
plt.show()


# ### Exercise:
# 
# Change x_index and y_index in the above script and find a combination of two parameters which maximally separate the three classes.

# ### 4.5. PCA
# 
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions

# In[3]:

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#PCA
pca = PCA(n_components=3)
result = pca.fit_transform(iris.data)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

#Plot 1D
plt.figure()
for c, i, iris.target_name in zip("rgb", [0,1,2], iris.target_names):
    plt.scatter(result[iris.target == i, 0], result[iris.target == i, 1], c=c, label=iris.target_name)
plt.legend()
plt.title('PCA of Iris dataset')
plt.show()

#Plot 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=iris.target)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


# ### 4.6. Classify the iris dataset
# 
# Scikits-learn has a very common interface for all it's models making it easy to use and switch between models. The two major stages are 1) fit where we fit a model, or learn from the data and 2) predict where we extrapolate from what we learned.
# 
# ### 4.6.1. KNN
# 
# The simplest possible classifier is the nearest neighbor: given a new observation, take the label of the training samples closest to it in n-dimensional space, where n is the number of features in each sample. The k-nearest neighbors classifier internally uses an algorithm based on ball trees to represent the samples it is trained on.
# 
# Scikits-learn has a very common interface for all it's models making it easy to use and switch between models. The two major stages are 1) fit where we fit a model, or learn from the data and 2) predict where we extrapolate from what we learned.

# In[22]:

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(iris.data, iris.target) 

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
predicted = knn.predict([[3, 5, 4, 2]])
print predicted 

#what are the names
print iris.target_names[predicted]

#more predictions
print iris.target_names[knn.predict([[0.1, 0.2, 0.3, 0.4],
                                     [7, 0.2, 4, 10],
                                     [3, 4, 5, 2]
                                     ])]

# # Further reading
# 
#     http://numpy.scipy.org
#     http://scipy.org/Tentative_NumPy_Tutorial
#     http://scipy.org/NumPy_for_Matlab_Users - A Numpy guide for MATLAB users.
#     http://scikit-learn.org/stable/supervised_learning.html#supervised-learning - scikit-learn supervised learning page
#     http://en.wikipedia.org/wiki/Iris_flower_data_set - More information on the Iris dataset
#     http://www.slideshare.net/SarahGuido/a-beginners-guide-to-machine-learning-with-scikitlearn - A tutorial to scikitlearn
# 
