
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
# Machine learning is a field of computer science that deals with the construction of algorithms that can learn from data and can be used to make predictions or decisions. Machine learning algorithms provide computers with the ability to search through data, look for patterns, and uses extracted information to make educated decisions without the computer being explicitly programmed with static instructions.
# 
# ### 1.2. Machine learning examples
# 
# Lets run through some examples of machine learning problems.
# 
# #### Recommender systems
# 
# Netflix is a movie rental company in the US. Netflix has a movie recommender system that changes according to the user's personal movie reantals. If a user frequently gives high ratings to comedy movies or TV shows, Netflix may want to recommend more comedy-related videos. So, one of the goals of recommender systems is to try and predict what the customers would think of the other movies based on what they rated so far. 

# In[7]:

from IPython.display import display, Image
display(Image(filename='images/movie_recommender.jpg'))


# Out[7]:

# image file:

# #### Spam filter
# 
# Another classical example is the email spam detection. In this case we want to build a classifier that will classify an email as "spam" or "not spam". The data in the word clouds below were taken from over thousands of "spam" or "good" emails. The goal of the classifier is to classify spam from email based on the frequency of words in the email. This is a typical text classification example 

# In[1]:

from IPython.display import display, Image
display(Image(filename='images/spam_wordcloud.png'))


# Out[1]:

# image file:

# ### Recognizing hand-written digits
# 
# This is another tipical example of machine learning classification procedure, it was one of the first learning tasks that was used to develop neural networks. Consider some handwritten digits below. The goal is to, based on an image of any of these digits, say what the digit is, to classify into the 10 digit classes.
# Well to humans, this looks like a pretty easy task. We're pretty good at pattern recognition. It turns out it's a really difficult task to crack for computers. They're getting better and better all the time.

# In[2]:

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


# Out[2]:

#     (1797, 64)
#     [  0.   0.   5.  13.   9.   1.   0.   0.   0.   0.  13.  15.  10.  15.   5.
#        0.   0.   3.  15.   2.   0.  11.   8.   0.   0.   4.  12.   0.   0.   8.
#        8.   0.   0.   5.   8.   0.   0.   9.   8.   0.   0.   4.  11.   0.   1.
#       12.   7.   0.   0.   2.  14.   5.  10.  12.   0.   0.   0.   0.   6.  13.
#       10.   0.   0.   0.]
#     [0 1 2 ..., 8 9 8]
#     (1797, 64)
#     (1797, 8, 8)
#     True
#     (4447547392, False)
#     (4447547392, False)
# 

# In[4]:

from IPython.display import display, Image
display(Image(filename='images/digits_figure.png'))


# Out[4]:

# image file:

# Each feature is a real-valued quantity representing the
# darkness of a pixel in an 8x8 image of a hand-written digit.
# 
# Even though each sample has data that is inherently two-dimensional, the data matrix flattens
# this 2D data into a **single vector**, which can be contained in one **row** of the data matrix.

# ### Face recognition
# 
# 
# ### Classifying tissue samples
# So the next example comes from medicine, classifying a tissue sample into one of several cancer classes based on sample features.

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
# ### Supervised and unsupervised problems
# 
# A classifier is a system that inputs (typically) a vector of discrete and/or continuous predictor values and outputs a single discrete value, the class. The output variable goes by various names: Y, dependent variable, response or target. The predictor measurments are also called inputs, regressors, covariates, features, or independent variables.
# 
# Machine learning tasks are typically classified into three broad categories, depending on the nature of the learning "signal" or "feedback" available to a learning system. These are:
# 
# - Supervised learning. The computer is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs.
# 
# - Unsupervised learning, no labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end.

# In[6]:

#Show scikit-learn cheat sheet: (http://peekaboo-vision.blogspot.co.uk/2013/01/machine-learning-cheat-sheet-for-scikit.html)
from IPython.display import display, Image
display(Image(filename='images/drop_shadows_background.png'))


# Out[6]:

# image file:

# The idea of supervised learning, in kindergarten of a teacher trying to teach a child to discriminate between what, say, a house is and a bike. So he might show the child some examples of what a house looks like. And here's some examples of what a bike looks like.
# He tells Johnny this and shows him examples of each of the classes. And the child then learns, oh I see. Houses got sort of square edges, and a bike has got some more rounded edges, et cetera. That's supervised learning, because he's been given examples of label training observations. He's been supervised. The Y there is given and the child tries to learn to classify the two objects based on the features, the X's. Finally we need to distinguish if the learner has made good predictions by evaluating the classifie (e.g., whether the spam filter correctly classifies previously unseen emails as spam or not spam).
# 
# In unsupervised learning, was not given examples of what a house and a bike was. He just sees on the ground lots of things. He sees maybe some houses, some bikes, some other things. And so this data is unlabeled. There's no Y.
# So the problem there now is for the child, it's unsupervised, to try to organize in his own mind the common patterns of what he sees.
# He may look at the objects and say, oh these three things are similar to each other because they have common features.
# These other objects, are similar to each other, because I see some commonality. And that brings the idea of trying to group observations by similarity of features.
# So more formally, there's no outcome variable measure, just a set of predictors. And the objective is more fuzzy. It's not just to predict Y, because there is no Y. It's rather to learn about how the data is organized, and to find which features are important for the organization of the data. Hierarchical clustering is an important technique for unsupervised learning. One of the challenges is it's hard to know how well you're doing. There's no gold standard. There's no Y. 
# But nonetheless, it's an extremely important area. One reason is that the idea of unsupervised learning is an important preprocessor for supervised learning. It's often useful to try to organize your features, choose features based on the X's themselves, and then use those processed or chosen features as input into supervised learning. And the last point is that it's a lot easier, it's a lot more common to collect data which is unlabeled. Because on the web, for example, if you look at movie
# reviews, a computer algorithm can just scan the web and grab reviews. Figuring out whether review, on the other hand, is positive
# or negative often takes human intervention. So it's much harder and costly to label data. Much easier just to collect unsupervised, unlabeled data.

# ### Quiz: Is this a suppervised or an unsupervised problem?
# 

### Learning activity 3: Scikit learn interface [30 min]

# Learning outcomes:
# 
# - Understand basics of data manipulation with numpy arrays
# 
# - Understand scikit-learn syntax
# 
# - Know how to load data into scikit-learn
# 
# - Understand data format and shape within scikit-learn

# ##  Data manipulation with Numpy with simple example exercises
# 
# Being able to manipulate numpy arrays is an important part of doing machine learning in Python. Numpy provides high-performance vector, matrix and higher-dimensional data structures for calculations in Python. 
# 
# In Numpy the terminology used for data sets is array. Numpy is implemented in C and Fortran, which makes computations with numpy arrays memory efficient
# 
# To use numpy need to import the module:

# In[8]:

import numpy as np


# ### Creating numpy arrays
# 
# There are a number of ways to create numpy arrays. For example

# In[21]:

# Generate a vector from a Python list 
X = np.array([1,2,3,4])
print X


# Out[21]:

#     [1 2 3 4]
# 

# In[22]:

# Generate a random array
R = np.random.random((10, 5))  # a 10 x 5 matrix
print R


# Out[22]:

#     [[ 0.38791387  0.32526028  0.14153182  0.71865651  0.18970824]
#      [ 0.19354106  0.71395833  0.58767     0.68926203  0.27404207]
#      [ 0.21817474  0.52303377  0.85382153  0.03815955  0.39870863]]
# 

# In[39]:

# Generate a matrix from a Python nested list
M = np.array([[1, 2], [3, 4]])
print M


# Out[39]:

#     [[1 2]
#      [3 4]]
# 

# ### Properties of numpy arrays
# 
# The shape and the size methods can be applied to numpy ndarray objects:

# In[42]:

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


# Out[42]:

#     Shape:
#     (4,)
#     (3, 5)
#     (2, 2)
#     Size:
#     4
#     15
#     4
#     np.shape and np.size:
#     (2, 2)
#     4
#     8
#     2
# 

# ### More ways to create numpy arrays
# 
# We can generate longer arrays automatically using functions, instead of entering the data manually with Python lists. Some of the more common are:

# In[38]:

#More ways to create numpy arrays

#Create a range
x = np.arange(0, 10, 1)
print x

x = np.arange(-1, 1, 0.1)
print x

# using linspace and logspace
print np.linspace(0, 100, 11)
print np.logspace(0, 100, 11, base=10)


# Out[38]:

#     [0 1 2 3 4 5 6 7 8 9]
#     [ -1.00000000e+00  -9.00000000e-01  -8.00000000e-01  -7.00000000e-01
#       -6.00000000e-01  -5.00000000e-01  -4.00000000e-01  -3.00000000e-01
#       -2.00000000e-01  -1.00000000e-01  -2.22044605e-16   1.00000000e-01
#        2.00000000e-01   3.00000000e-01   4.00000000e-01   5.00000000e-01
#        6.00000000e-01   7.00000000e-01   8.00000000e-01   9.00000000e-01]
#     [   0.   10.   20.   30.   40.   50.   60.   70.   80.   90.  100.]
#     [  1.00000000e+000   1.00000000e+010   1.00000000e+020   1.00000000e+030
#        1.00000000e+040   1.00000000e+050   1.00000000e+060   1.00000000e+070
#        1.00000000e+080   1.00000000e+090   1.00000000e+100]
# 

# ### Manipulating arrays
# 
# We can index elements in an array using the square bracket and indices:

# In[71]:

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


# Out[71]:

#     [[ 0.89847285  0.60576358  0.26342986  0.72932682  0.38711389]
#      [ 0.37799779  0.88391218  0.99682069  0.83695834  0.91059773]
#      [ 0.56084463  0.83667455  0.00575926  0.3639832   0.21046016]]
#     0.898472850998
#     [ 0.37799779  0.88391218  0.99682069  0.83695834  0.91059773]
#     row:   0.377998   0.883912   0.996821   0.836958   0.910598
#     [ 0.60576358  0.88391218  0.83667455]
#     Column:   0.605764   0.883912   0.836675
#     2
#     single element of a vector : 2
# 

# ### Index slicing
# 
# Index slicing is the technical name for the syntax M[lower:upper:step] to extract part of an array:

# In[90]:

A = np.array([1,2,3,4,5])

# elements from 1 to 3
print A[1:3]

# first three elements 
print A[:3] 

# elements from index 3
print A[3:] 


# Out[90]:

#     [2 3]
#     [1 2 3]
#     [4 5]
# 

# Negative indices counts from the end of the array (positive index from the begining):

# In[95]:

# last element
print A[-1] 

# the last three elements
print A[-3:] 


# Out[95]:

#     [40 41 42 43 44]
#     [[20 21 22 23 24]
#      [30 31 32 33 34]
#      [40 41 42 43 44]]
# 

# Slicing applies the same way to multidimensional arrays

# In[96]:

A = np.array([[n+m*10 for n in range(5)] for m in range(5)])
print A

# get a block from the original array
print A[1:4, 1:4]


# Out[96]:

#     [[ 0  1  2  3  4]
#      [10 11 12 13 14]
#      [20 21 22 23 24]
#      [30 31 32 33 34]
#      [40 41 42 43 44]]
#     [[11 12 13]
#      [21 22 23]
#      [31 32 33]]
# 

# ### Assign values to arrays
# 
# We can assign new values to elements in an array using indexing:

# In[66]:

M = np.array([[1, 2], [3, 4]])
print M


# Out[66]:

#     [[1 2]
#      [3 4]]
# 

# In[67]:

M[0,0] = 1100
print M


# Out[67]:

#     [[1100    2]
#      [   3    4]]
# 

# In[68]:

# also works for rows and columns
M[1,:] = 0
print M


# Out[68]:

#     [[1100    2]
#      [   0    0]]
# 

# In[69]:

#also works for 1D arrays
v = np.array([1,2,3,4])
v[0] = 30
print v


# Out[69]:

#     [30  2  3  4]
# 

# In[70]:

#Create a sparce matrix (an array with lots of zeros)
#Replace values smaller than 0.7
X = np.random.random((10,5))
X[X < 0.7] = 0
print X


# Out[70]:

#     [[ 0.          0.          0.          0.          0.        ]
#      [ 0.          0.          0.          0.          0.83901515]
#      [ 0.          0.          0.          0.94267514  0.        ]
#      [ 0.          0.78652884  0.82929059  0.97230065  0.        ]
#      [ 0.96569135  0.          0.          0.87899507  0.        ]
#      [ 0.          0.85387237  0.          0.          0.        ]
#      [ 0.77466504  0.          0.          0.          0.        ]
#      [ 0.73269898  0.80778151  0.          0.          0.        ]
#      [ 0.          0.7707395   0.          0.          0.        ]
#      [ 0.87037053  0.79619028  0.82494542  0.          0.        ]]
# 

# We get an error if we try to assign the wrong data type to an array

# In[73]:

#Using the dtype we can see what type the data of an array has
X.dtype

#Try to assign a string to the wrong type
M[0,0] = "hello"


# Out[73]:


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-73-6fde34bc5a77> in <module>()
          3 
          4 #Try to assign a string to the wrong type
    ----> 5 M[0,0] = "hello"
    

    ValueError: invalid literal for long() with base 10: 'hello'


# If we want, we can explicitly define the type of the array data when we create it, using the dtype keyword argument. Common type that can be used with dtype are: int, float, complex, bool, object. We can also explicitly define the bit size of the data types, for example: int64, int16, float128, complex128.

# In[88]:

M = np.array([[1, 2], [3, 14]], dtype=complex)
print M

M = np.array([["hello", 2], [3, 14]], dtype=str)
print M
M[0,0] = 'inesinha'
print M


# Out[88]:

#     [[  1.+0.j   2.+0.j]
#      [  3.+0.j  14.+0.j]]
#     [['hello' '2']
#      ['3' '14']]
#     [['inesi' '2']
#      ['3' '14']]
# 

# In[79]:

# Transposing an array
X = np.random.random((3,5))
print X
print X.T


# Out[79]:

#     [[ 0.95213229  0.47237824  0.08949783  0.27099196  0.0086144 ]
#      [ 0.67215123  0.84953544  0.57780725  0.36871919  0.08161305]
#      [ 0.20596178  0.4243674   0.88596755  0.56235655  0.72060864]]
#     [[ 0.95213229  0.67215123  0.20596178]
#      [ 0.47237824  0.84953544  0.4243674 ]
#      [ 0.08949783  0.57780725  0.88596755]
#      [ 0.27099196  0.36871919  0.56235655]
#      [ 0.0086144   0.08161305  0.72060864]]
# 

# In[51]:

# Turning a row vector into a column vector
y = np.linspace(0, 100, 5)

print y

# make into a column vector
print y[:, np.newaxis]


# Out[51]:

#     [   0.   25.   50.   75.  100.]
#     [[   0.]
#      [  25.]
#      [  50.]
#      [  75.]
#      [ 100.]]
# 

# ## Representation of Data in Scikit-learn
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

# ## Load an example dataset using scikit-learn: The iris dataset
# 
# As an example of a simple dataset, we're going to take a look at the
# iris data stored by scikit-learn.
# The data consists of measurements of three different species of irises.
# There are three species of iris in the dataset, which we can picture here:
# 
# scikits.learn has become one of the most popular machine learning packages in Python. It has a most of the standard machine learning algorithms built-in and often calls faster external C-code that can make execuation fairly quick.

# In[15]:

from IPython.core.display import Image, display
display(Image(filename='images/iris_setosa.jpg'))
print "Iris Setosa\n"

display(Image(filename='images/iris_versicolor.jpg'))
print "Iris Versicolor\n"

display(Image(filename='images/iris_virginica.jpg'))
print "Iris Virginica"


# Out[15]:

# image file:

#     Iris Setosa
#     
# 

# image file:

#     Iris Versicolor
#     
# 

# image file:

#     Iris Virginica
# 

# ### Quiz Question:
# 
# **If we want to design an algorithm to recognize iris species, what might the features be? What might the labels be?**
# 
# Remember: we need a 2D data array of size `[n_samples x n_features]`, and a 1D label array of size `n_samples`.
# 
# - What would the `n_samples` refer to?
# 
# - What might the `n_features` refer to?
# 
# Remember that there must be a **fixed** number of features for each sample, and feature
# number ``i`` must be a similar kind of quantity for each sample.

# ### Loading the Iris Data with Scikit-Learn
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

# In[122]:

from sklearn.datasets import load_iris
iris = load_iris()


# This load in one of most well-known toy datasets for machine learning. It's useful since it's very easy to get good performance on. Each row contains one flower instance with information on it's petal and sepal measurements. It's flower is classified into 1 of 3 species.
# 
# The result is a ``Bunch()`` object, which is basically an enhanced dictionary which contains the data.
# The ``Bunch()`` object, Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘target’, the classification labels, ‘target_names’, the meaning of the labels, ‘feature_names’, the meaning of the features, and ‘DESCR’, the full description of the dataset.
# 
# **Note that bunch objects are not required for performing learning in scikit-learn, they are simply a convenient container for the numpy arrays which *are* required**

# The iris dataset is a classic and very easy multi-class classification dataset.
# Classes 	3
# Samples per class 	50
# Samples total 	150
# Dimensionality 	4
# Features 	real, positive

# This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
# 
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

# In[123]:

iris.keys()


# Out[123]:

#     ['target_names', 'data', 'target', 'DESCR', 'feature_names']

# In[103]:

n_samples, n_features = iris.data.shape
print (n_samples, n_features)
print iris.data[:10]


# Out[103]:

#     (150, 4)
#     [[ 5.1  3.5  1.4  0.2]
#      [ 4.9  3.   1.4  0.2]
#      [ 4.7  3.2  1.3  0.2]
#      [ 4.6  3.1  1.5  0.2]
#      [ 5.   3.6  1.4  0.2]
#      [ 5.4  3.9  1.7  0.4]
#      [ 4.6  3.4  1.4  0.3]
#      [ 5.   3.4  1.5  0.2]
#      [ 4.4  2.9  1.4  0.2]
#      [ 4.9  3.1  1.5  0.1]]
# 

# Let’s say you are interested in the samples 10, 25, and 50, and want to know their class name.

# In[104]:

iris.target[[50,20]]


# Out[104]:

#     array([1, 0])

# In[106]:

print iris.target
print iris.target[45:55]
print iris.target_names[iris.target[45:55]] #an array was used in-place of an index


# Out[106]:

#     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#      0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#      2 2]
#     [0 0 0 0 0 1 1 1 1 1]
#     ['setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'versicolor' 'versicolor'
#      'versicolor' 'versicolor' 'versicolor']
# 

# In[107]:

print iris.data.shape
print iris.target.shape


# Out[107]:

#     (150, 4)
#     (150,)
# 

# In[108]:

print iris.target


# Out[108]:

#     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#      0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#      2 2]
# 

# In[121]:

print iris.target_names


# Out[121]:

#     ['setosa' 'versicolor' 'virginica']
# 

# In[120]:

import matplotlib.pyplot as plt

X = iris.data  # we only take the first two features.
print X[:10,]
Y = iris.target
y


# Out[120]:

#     [[ 5.1  3.5  1.4  0.2]
#      [ 4.9  3.   1.4  0.2]
#      [ 4.7  3.2  1.3  0.2]
#      [ 4.6  3.1  1.5  0.2]
#      [ 5.   3.6  1.4  0.2]
#      [ 5.4  3.9  1.7  0.4]
#      [ 4.6  3.4  1.4  0.3]
#      [ 5.   3.4  1.5  0.2]
#      [ 4.4  2.9  1.4  0.2]
#      [ 4.9  3.1  1.5  0.1]]
# 

#     array([   0.,   25.,   50.,   75.,  100.])

# This data is four dimensional, but we can visualize two of the dimensions at a time using a simple scatter-plot:

# In[128]:

get_ipython().magic(u'matplotlib inline')

#x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#plt.figure(2, figsize=(8, 6))
#plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')


# Out[128]:

#     <matplotlib.text.Text at 0x10890f4d0>

# image file:

# In[129]:

get_ipython().magic(u'matplotlib inline')

# Plot the training points
plt.scatter(X[:, 2], X[:, 3], c=Y)
plt.xlabel('Petal length')
plt.ylabel('Petal width')


# Out[129]:

#     <matplotlib.text.Text at 0x111549d90>

# image file:

# ### Exercise:
# 
# Change x_index and y_index in the above script and find a combination of two parameters which maximally separate the three classes.

# ## Classify the iris dataset
# 
# 

# ### KNN

# Scikits-learn has a very common interface for all it's models making it easy to use and switch between models. The two major stages are 1) fit where we fit a model, or learn from the data and 2) predict where we extrapolate from what we learned.

# In[89]:

from sklearn import neighbors
X, y = iris.data, iris.target
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
print iris.target_names[knn.predict([[3, 5, 4, 2]])]


# Out[89]:

#     ['virginica']
# 

# ### SVM

# ### Cross validation

# In[56]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)


# Out[56]:


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-56-52b714b9757c> in <module>()
          1 from sklearn.cross_validation import train_test_split
    ----> 2 X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
          3 
          4 knn = neighbors.KNeighborsClassifier(n_neighbors=1)
          5 knn.fit(X_train, y_train)


    NameError: name 'cross_validation' is not defined


# In[57]:

from sklearn import metrics
print metrics.classification_report(y_test, predicted)
print metrics.confusion_matrix(y_test, predicted)
print metrics.f1_score(y_test, predicted)


# Out[57]:


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-57-b61916a5693b> in <module>()
          1 from sklearn import metrics
    ----> 2 print metrics.classification_report(y_test, predicted)
          3 print metrics.confusion_matrix(y_test, predicted)
          4 print metrics.f1_score(y_test, predicted)


    NameError: name 'y_test' is not defined


# In[58]:

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(knn, iris.data, iris.target, cv=5)


# Out[58]:

#     /Users/santia01/anaconda/lib/python2.7/site-packages/sklearn/neighbors/classification.py:131: NeighborsWarning: kneighbors: neighbor k+1 and neighbor k have the same distance: results will be dependent on data order.
#       neigh_dist, neigh_ind = self.kneighbors(X)
#     /Users/santia01/anaconda/lib/python2.7/site-packages/sklearn/neighbors/classification.py:131: NeighborsWarning: kneighbors: neighbor k+1 and neighbor k have the same distance: results will be dependent on data order.
#       neigh_dist, neigh_ind = self.kneighbors(X)
#     /Users/santia01/anaconda/lib/python2.7/site-packages/sklearn/neighbors/classification.py:131: NeighborsWarning: kneighbors: neighbor k+1 and neighbor k have the same distance: results will be dependent on data order.
#       neigh_dist, neigh_ind = self.kneighbors(X)
# 

# In[ ]:




# # Further reading
# 
#     http://numpy.scipy.org
#     http://scipy.org/Tentative_NumPy_Tutorial
#     http://scipy.org/NumPy_for_Matlab_Users - A Numpy guide for MATLAB users.
#     http://scikit-learn.org/stable/supervised_learning.html#supervised-learning - scikit-learn supervised learning page
