
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 38
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
# FORWARD PROPAGATION (FROM X TO COST)

# Normalize/Starndardize Dataset - in this case since image data - simply deviding by the max value of RGB range will suffice.
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# Mathematical expression of the algorithm:

# For one example x(i)x(i):
# z(i)=wTx(i)+b(1)
# (1)z(i)=wTx(i)+b
# ŷ (i)=a(i)=sigmoid(z(i))(2)
# (2)y^(i)=a(i)=sigmoid(z(i))
# (a(i),y(i))=−y(i)log(a(i))−(1−y(i))log(1−a(i))(3)
# (3)L(a(i),y(i))=−y(i)log⁡(a(i))−(1−y(i))log⁡(1−a(i))
# The cost is then computed by summing over all training examples:
# J=1m∑i=1m(a(i),y(i))(6)
# (6)J=1m∑i=1mL(a(i),y(i))

# This code will carry out the following key steps:

# - Initialize the parameters of the model
# - Learn the parameters for the model by minimizing the cost
# - Use the learned parameters to make predictions (on the test set)
# - Analyse the results and conclude

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-z))
    ### END CODE HERE ###

    return s
