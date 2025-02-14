import random

import matplotlib.pyplot as plt
import sys

import numpy
import numpy as np
import scipy.io

########## [ SETUP ] #################################################################################

## loading the .mat file containing the images and x- and y coords.
file_path = "COIL20.mat"
mat = scipy.io.loadmat(file_path)
x_coords: numpy.ndarray = mat['X']
y_coords: numpy.ndarray = mat['Y']

########## [ INPUT 1 ] #################################################################################

# output is a tuple of 3 arrays
def pca(x: numpy.ndarray, d: int) -> tuple:
    mean = x.mean()
    print("mean =", mean)
    st_dev = x.std()
    print("st_dev =", st_dev)

    Z: numpy.ndarray = (x - mean)/st_dev
    print("\nZ:", Z)
    Z.resize((1024, 1024))
    print("\nZ resized to 1024x1024:", Z)
    D = np.linalg.eig(Z)
    print("\nD (eigenvalues of Z):", D)

    # return a matrix containing principal components Ud, a matrix (or a vector)
    # containing eigen-values, and reduced version of the data set Zd
    # return (Ud, D, Zd)

pca(x_coords, 0)

########## [ INPUT 2 ] #################################################################################

## Function may take an int (sample_index) as an argument, and will display the sample with that index.
## Otherwise, it will display a random sample. If a given sample_index is too high or low, it will
## display the sample with either the highest or lowest possible index respectively.
def input_data_sample(sample_index: int = None):
    if sample_index is None:
        sample_index = random.randrange(0, 1439)
    elif sample_index > 1439:
        sample_index = 1439
    elif sample_index < 0:
        sample_index = 0

    image_index = x_coords[sample_index, :]
    image = np.reshape(image_index, (-1, 32))

    plt.imshow(image, origin='lower')
    plt.title("Input data sample as an image")
    plt.show()

#input_data_sample()

########## [ INPUT 3 ] #################################################################################

def eigen_value_profile(x: numpy.ndarray, d: int):
    mean = x.mean()
    st_dev = x.std()
    Z: numpy.ndarray = (x - mean)/st_dev
    #eigen_tuple = np.linalg.eig(Z)


########## [ INPUT 4 ] #################################################################################

def dimension_reduced_data(x: numpy.array(numpy.array(float)), y: numpy.array(float),
                           d: int, perplexity: int, random_state: int):
    pass

#plt.savefig(sys.stdout.buffer)