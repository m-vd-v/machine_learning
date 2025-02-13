import random

import matplotlib.pyplot as plt
import sys

import numpy
import numpy as np
import scipy.io

file_path = "COIL20.mat"
mat = scipy.io.loadmat(file_path)
x_coords: numpy.ndarray = mat['X']
y_coords: numpy.ndarray = mat['Y']

# input 1
# output is 2 arrays
def pca(x: numpy.ndarray, d: int) -> str:
    mean = x.mean()
    st_dev = x.std()

    Z: numpy.ndarray = (x - mean)/st_dev
    print(Z)

    eigen_tuple = np.linalg.eig(Z)

    print(mean)
    print(st_dev)
    print(eigen_tuple)
    pass

#pca(x_coords, 0)


# input 2
def input_data_sample():
    rng = random.randrange(0, 1439)
    x1 = x_coords[rng, :]
    print(rng)

    B = np.reshape(x1, (-1, 32))

    fig, ax = plt.subplots()
    ax.imshow(B, origin='lower')
    plt.show()

# input 3
def eigen_value_profile(x: numpy.ndarray, d: int):
    mean = x.mean()
    st_dev = x.std()
    Z: numpy.ndarray = (x - mean)/st_dev
    #eigen_tuple = np.linalg.eig(Z)


# input 4
def dimension_reduced_data(x: numpy.array(numpy.array(float)), y: numpy.array(float),
                           d: int, perplexity: int, random_state: int):
    pass

plt.savefig(sys.stdout.buffer)

input_data_sample()