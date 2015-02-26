
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

# In[9]:

from IPython.display import display, Image
display(Image(filename='images/movie_recommender.jpg'))


# Out[9]:

# image file:

# #### Spam filter
# 
# Another classical example is the email spam detection. In this case we want 
# to build a classifier that will classify an email as "spam" or "not spam". 
# The data in the word clouds below were taken from over thousands of "spam" or "good" emails. 
# The goal of the classifier is to classify spam from email based on the frequency of words in the email. 
# This is a typical text classification example 

# In[10]:

from IPython.display import display, Image
display(Image(filename='images/spam_wordcloud.png'))


# Out[10]:

# image file:

# ### Recognizing hand-written digits
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


# Out[11]:

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
#     (4455399424, False)
#     (4455399424, False)
# 

# In[12]:

from IPython.display import display, Image
display(Image(filename='images/digits_figure.png'))


# Out[12]:

# image file:

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
# ### Supervised and unsupervised problems
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

# In[13]:

#Show scikit-learn cheat sheet: (http://peekaboo-vision.blogspot.co.uk/2013/01/machine-learning-cheat-sheet-for-scikit.html)
from IPython.display import display, Image
display(Image(filename='images/drop_shadows_background.png'))


# Out[13]:

# image file:

# Unsupervised learning is an important preprocessor for supervised learning. 
# It's often useful to try to organize your features, choose features based on the X's themselves, 
# and then use those processed or chosen features as input into supervised learning. 
# It is also a lot more common to collect data which is unlabeled. 
# 
# ### Trainning vs discovery dataset
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

# #Learning activity 3: Scikit learn interface [30 min] 
# 
# Learning outcomes:
# 
# - Understand basics of data manipulation with numpy arrays
# 
# - Understand scikit-learn syntax
# 
# - Know how to load data into scikit-learn
# 
# - Understand data format and shape within scikit-learn
# 
# ##  Data manipulation with Numpy with simple example exercises
# 
# Being able to manipulate numpy arrays is an important part of doing machine learning in Python. Numpy provides high-performance vector, matrix and higher-dimensional data structures for calculations in Python. 
# 
# In Numpy the terminology used for data sets is array. Numpy is implemented in C and Fortran, which makes computations with numpy arrays memory efficient
# 
# To use numpy need to import the module:

# In[14]:

import numpy as np


# ### Creating numpy arrays
# 
# There are a number of ways to create numpy arrays. For example

# In[15]:

# Generate a vector from a Python list 
X = np.array([1,2,3,4])
print X


# Out[15]:

#     [1 2 3 4]
# 

# In[16]:

# Generate a random array
R = np.random.random((10, 5))  # a 10 x 5 matrix
print R


# Out[16]:

#     [[ 0.96673763  0.2130024   0.28761298  0.13796016  0.17302061]
#      [ 0.74725261  0.85271765  0.91141793  0.45485285  0.70213601]
#      [ 0.28078397  0.8613403   0.54537679  0.29526851  0.71799205]
#      [ 0.3988181   0.50601354  0.73678075  0.65962224  0.41972515]
#      [ 0.38115451  0.79286497  0.62552811  0.18160768  0.01581078]
#      [ 0.83570284  0.08333593  0.49199635  0.29631227  0.15591427]
#      [ 0.79351297  0.43624988  0.79053603  0.55551422  0.72260809]
#      [ 0.49691926  0.17388796  0.17247526  0.21384287  0.67159634]
#      [ 0.78009506  0.23305675  0.03384673  0.36730243  0.86913374]
#      [ 0.1673632   0.87108288  0.39531035  0.64140693  0.13980584]]
# 

# In[17]:

# Generate a matrix from a Python nested list
M = np.array([[1, 2], [3, 4]])
print M


# Out[17]:

#     [[1 2]
#      [3 4]]
# 

# ### Properties of numpy arrays
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


# Out[18]:

#     Shape:
#     (4,)
#     (10, 5)
#     (2, 2)
#     Size:
#     4
#     50
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


# Out[19]:

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


# Out[20]:

#     [[ 0.74742437  0.58253544  0.03546311  0.57790876  0.96527153]
#      [ 0.13218282  0.2117342   0.45278113  0.78564846  0.32065856]
#      [ 0.80818378  0.33522893  0.90458834  0.75044079  0.55208418]]
#     0.747424372487
#     [ 0.13218282  0.2117342   0.45278113  0.78564846  0.32065856]
#     row:   0.132183   0.211734   0.452781   0.785648   0.320659
#     [ 0.58253544  0.2117342   0.33522893]
#     Column:   0.582535   0.211734   0.335229
#     2
#     single element of a vector : 2
# 

# ### Index slicing
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


# Out[21]:

#     [2 3]
#     [1 2 3]
#     [4 5]
# 

# Negative indices counts from the end of the array (positive index from the begining):

# In[22]:

# last element
print A[-1] 

# the last three elements
print A[-3:] 


# Out[22]:

#     5
#     [3 4 5]
# 

# Slicing applies the same way to multidimensional arrays

# In[23]:

A = np.array([[n+m*10 for n in range(5)] for m in range(5)])
print A

# get a block from the original array
print A[1:4, 1:4]


# Out[23]:

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

# In[24]:

M = np.array([[1, 2], [3, 4]])
print M


# Out[24]:

#     [[1 2]
#      [3 4]]
# 

# In[25]:

M[0,0] = 1100
print M


# Out[25]:

#     [[1100    2]
#      [   3    4]]
# 

# In[26]:

# also works for rows and columns
M[1,:] = 0
print M


# Out[26]:

#     [[1100    2]
#      [   0    0]]
# 

# In[27]:

#also works for 1D arrays
v = np.array([1,2,3,4])
v[0] = 30
print v


# Out[27]:

#     [30  2  3  4]
# 

# In[28]:

#Create a sparce matrix (an array with lots of zeros)
#Replace values smaller than 0.7
X = np.random.random((10,5))
X[X < 0.7] = 0
print X


# Out[28]:

#     [[ 0.          0.78187846  0.          0.          0.        ]
#      [ 0.          0.          0.          0.83904368  0.99835835]
#      [ 0.          0.          0.82948543  0.          0.        ]
#      [ 0.96471726  0.          0.          0.          0.84235955]
#      [ 0.          0.          0.97234209  0.          0.76122466]
#      [ 0.          0.          0.82349762  0.73373833  0.        ]
#      [ 0.          0.96470602  0.          0.          0.79832002]
#      [ 0.86191236  0.          0.87785347  0.          0.92161389]
#      [ 0.88084492  0.84486546  0.          0.95243958  0.        ]
#      [ 0.          0.77730676  0.          0.          0.        ]]
# 

# We get an error if we try to assign the wrong data type to an array

# In[29]:

#Using the dtype we can see what type the data of an array has
X.dtype

#Try to assign a string to the wrong type
M[0,0] = "hello"


# Out[29]:


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-29-6fde34bc5a77> in <module>()
          3 
          4 #Try to assign a string to the wrong type
    ----> 5 M[0,0] = "hello"
    

    ValueError: invalid literal for long() with base 10: 'hello'


# If we want, we can explicitly define the type of the array data when we create it, using the dtype keyword argument. Common type that can be used with dtype are: int, float, complex, bool, object. We can also explicitly define the bit size of the data types, for example: int64, int16, float128, complex128.

# In[30]:

M = np.array([[1, 2], [3, 14]], dtype=complex)
print M


# Out[30]:

#     [[  1.+0.j   2.+0.j]
#      [  3.+0.j  14.+0.j]]
# 

# In[31]:

# Transposing an array
X = np.random.random((3,5))
print X
print X.T


# Out[31]:

#     [[ 0.25332356  0.78561226  0.46216555  0.90728557  0.05107888]
#      [ 0.18390764  0.77951414  0.29671559  0.15307562  0.73360408]
#      [ 0.24997955  0.22796432  0.67865686  0.39685587  0.75265091]]
#     [[ 0.25332356  0.18390764  0.24997955]
#      [ 0.78561226  0.77951414  0.22796432]
#      [ 0.46216555  0.29671559  0.67865686]
#      [ 0.90728557  0.15307562  0.39685587]
#      [ 0.05107888  0.73360408  0.75265091]]
# 

# In[32]:

# Turning a row vector into a column vector
y = np.linspace(0, 100, 5)

print y

# make into a column vector
print y[:, np.newaxis]


# Out[32]:

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


# Out[33]:

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
# We will need a matrix of `[samples x features]`, and a vector `samples`.
# 
# - What would the `samples` refer to?
# 
# - What might the `features` refer to?
# 
# - What are the X (predictor) and Y (target/response) variables? 

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

# In[34]:

from sklearn.datasets import load_iris
iris = load_iris()


# The result is the ``iris`` object, which is basically an dictionary-like object which contains the data.
# Few interesting attributes are: 

# In[37]:

iris.keys()


# Out[37]:

#     ['target_names', 'data', 'target', 'DESCR', 'feature_names']

# - ‘data’, a (samples x features) array with the data to learn 
# - ‘target’, the classification labels
# - ‘target_names’, the meaning of the labels
# - ‘feature_names’, the meaning of the features
# - ‘DESCR’, the full description of the dataset.

# ``iris.data`` gives you the features that can be used to classify the iris sample:

# In[42]:

#subset of first 10 rows:
print iris.data[:10,]


# Out[42]:

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

# This iris data is stored in the ``.data member``, which is a n_samples, n_features array. Each row contains one flower instance with information on it's petal and sepal measurments, stored in a 150x4 ``numpy.ndarray`` 

# In[43]:

#samples x features
n_samples, n_features = iris.data.shape
print (n_samples, n_features)


# Out[43]:

#     (150, 4)
# 

# The feature names correspond to the petal and septal measurments and are given by ``.feature_names``:

# In[47]:

#meaning of the features
print iris.feature_names


# Out[47]:

#     ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# 

# Each flower is labled with 3 different types of irises’ (Setosa, Versicolour, and Virginica). The labels are stored in the ``.target`` member or the iris dataset: 

# In[48]:

#target (numeric)
print iris.target

#shape of arrays
print iris.target.shape


# Out[48]:

#     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#      0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#      2 2]
#     (150,)
# 

# The names corresponding to each target feature is given by ``.target_names``. 

# In[44]:

#meaning of the target labels 
print iris.target_names
print iris.target_names[iris.target[:10]] #an array was used in-place of an index


# Out[44]:

#     ['setosa' 'versicolor' 'virginica']
#     ['setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'setosa'
#      'setosa' 'setosa']
# 

# In[46]:

#Full description of the dataset
print iris.DESCR


# Out[46]:

#     Iris Plants Database
#     
#     Notes
#     -----
#     Data Set Characteristics:
#         :Number of Instances: 150 (50 in each of three classes)
#         :Number of Attributes: 4 numeric, predictive attributes and the class
#         :Attribute Information:
#             - sepal length in cm
#             - sepal width in cm
#             - petal length in cm
#             - petal width in cm
#             - class:
#                     - Iris-Setosa
#                     - Iris-Versicolour
#                     - Iris-Virginica
#         :Summary Statistics:
#         ============== ==== ==== ======= ===== ====================
#                         Min  Max   Mean    SD   Class Correlation
#         ============== ==== ==== ======= ===== ====================
#         sepal length:   4.3  7.9   5.84   0.83    0.7826
#         sepal width:    2.0  4.4   3.05   0.43   -0.4194
#         petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
#         petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
#         ============== ==== ==== ======= ===== ====================
#         :Missing Attribute Values: None
#         :Class Distribution: 33.3% for each of 3 classes.
#         :Creator: R.A. Fisher
#         :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
#         :Date: July, 1988
#     
#     This is a copy of UCI ML iris datasets.
#     http://archive.ics.uci.edu/ml/datasets/Iris
#     
#     The famous Iris database, first used by Sir R.A Fisher
#     
#     This is perhaps the best known database to be found in the
#     pattern recognition literature.  Fisher's paper is a classic in the field and
#     is referenced frequently to this day.  (See Duda & Hart, for example.)  The
#     data set contains 3 classes of 50 instances each, where each class refers to a
#     type of iris plant.  One class is linearly separable from the other 2; the
#     latter are NOT linearly separable from each other.
#     
#     References
#     ----------
#        - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
#          Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
#          Mathematical Statistics" (John Wiley, NY, 1950).
#        - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
#          (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
#        - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
#          Structure and Classification Rule for Recognition in Partially Exposed
#          Environments".  IEEE Transactions on Pattern Analysis and Machine
#          Intelligence, Vol. PAMI-2, No. 1, 67-71.
#        - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
#          on Information Theory, May 1972, 431-433.
#        - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
#          conceptual clustering system finds 3 classes in the data.
#        - Many, many more ...
#     
# 

# ### Ploting the iris dataset

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
# Scikits-learn has a very common interface for all it's models making it easy to use and switch between models. The two major stages are 1) fit where we fit a model, or learn from the data and 2) predict where we extrapolate from what we learned.
# 
# ### KNN
# 
# The simplest possible classifier is the nearest neighbor: given a new observation, take the label of the training samples closest to it in n-dimensional space, where n is the number of features in each sample. The k-nearest neighbors classifier internally uses an algorithm based on ball trees to represent the samples it is trained on.

# In[68]:

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
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


# Out[68]:

#     [1]
#     ['versicolor']
#     ['setosa' 'virginica' 'versicolor']
# 

# When experimenting with learning algorithms, it is important not to test the 
# prediction of an estimator on the data used to fit the estimator. Indeed, 
# with the kNN estimator, we would always get perfect prediction on the training set.

# In[69]:

perm = np.random.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
knn.fit(iris.data[:100], iris.target[:100]) 
knn.score(iris.data[100:], iris.target[100:]) 


# Out[69]:

#     /Users/santia01/anaconda/lib/python2.7/site-packages/sklearn/neighbors/classification.py:131: NeighborsWarning: kneighbors: neighbor k+1 and neighbor k have the same distance: results will be dependent on data order.
#       neigh_dist, neigh_ind = self.kneighbors(X)
# 

#     0.97999999999999998

# ### Quiz question:
# Why did we use a random permutation?

# # Further reading
# 
#     http://numpy.scipy.org
#     http://scipy.org/Tentative_NumPy_Tutorial
#     http://scipy.org/NumPy_for_Matlab_Users - A Numpy guide for MATLAB users.
#     http://scikit-learn.org/stable/supervised_learning.html#supervised-learning - scikit-learn supervised learning page
#     http://en.wikipedia.org/wiki/Iris_flower_data_set - More information on the Iris dataset
