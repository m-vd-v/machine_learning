import random
from xml.etree.ElementTree import tostring

import matplotlib.pyplot as plt
import sys

import numpy
import numpy as np
import scipy.io
import sklearn
from sklearn.manifold import TSNE

########## [ SETUP ] #################################################################################

## loading the .mat file containing the images and x- and y coords.
file_path = "COIL20.mat"
mat = scipy.io.loadmat(file_path)
x_coords: numpy.ndarray = mat['X']
y_coords: numpy.ndarray = mat['Y']

########## [ INPUT 1 ] #################################################################################

def get_eigen(x: numpy.ndarray):
    x = x.T  # transposing the array, so we can easily iterate over columns instead of rows
    for i, column in enumerate(x):
        mean = column.mean()
        st_dev = column.std()
        x[i] = (x[i] - mean) / st_dev
        ## The mean and st_dev of each column are 0 and 1 respectively (accounting for rounding errors)
    Z = x.T  # transposing again to original state to get standardized array data set Z
    covar = np.cov(Z, rowvar=False)
    eigen_values: numpy.ndarray
    eigen_vectors: numpy.ndarray
    eigen_values, eigen_vectors = numpy.linalg.eig(covar)
    idx = eigen_values.argsort()[::-1]  # sorting and then reversing array to get an array sorted from high to low
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return (eigen_values, eigen_vectors, Z)

# output is a tuple of 3 arrays
def pca(x: numpy.ndarray[numpy.ndarray], d: int, debug: bool = False) -> tuple:
    eigen_values, eigen_vectors, Z = get_eigen(x)
    Ud = eigen_vectors[:, :d]   # calculating principal components while only keeping the first d components
    if debug:
        print("sorted eigen_values:", eigen_values)
        print("eigen_vectors moved along with corresponding eigen_values:", eigen_vectors)
        print("principal components Ud:", Ud)
        print("shape of Ud", Ud.shape)
    Zd = Z @ Ud
    if debug:
        print("Zd:", Zd)

    return (Ud, eigen_values, Zd)
    # return a matrix containing principal components Ud, a matrix (or a vector)
    # containing eigen-values, and reduced version of the data set Zd
    # return (Ud, D, Zd)

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
    plt.savefig(sys.stdout.buffer)
    plt.show()

#input_data_sample()

########## [ INPUT 3 ] #################################################################################

def eigen_value_profile(x: numpy.ndarray, d: int):
    eigen_values, _, _ = get_eigen(x)
    # sorted eigenvalues are correct when comparing with output in themis
    fig, ax = plt.subplots()
    ax.plot(eigen_values,"purple")
    ax.set(xlabel="Index eigen-value", ylabel="Eigen-value", title="Eigen-value Profile of the Dataset")
    plt.show()

#eigen_value_profile(x_coords, 40)


########## [ INPUT 4 ] #################################################################################

def dimension_reduced_data(x: numpy.array(numpy.array(float)), y: numpy.array(float),
                           d: int, perplexity: int, random_state: int):
    _, _, x_reduced = pca(x, d)
    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    x_tsne = tsne.fit_transform(x_reduced)
    # plot
    fig, ax = plt.subplots()
    start = 0
    end = 0
    for i in range(0,20):
        while y[end] == i:
            end = end + 1
        ax.scatter(x_tsne[start:(end-1), 0], x_tsne[start:(end-1),1], s=20, label="class " + str(i+1))
        start = end
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="t-SNE visualisation of dimension reduced data")
    plt.legend(title="Object ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

def min_dimensions(keep: float, x: numpy.ndarray):
    eigen_values, _, _ = get_eigen(x)
    all_eig = sum(eigen_values)
    store = 0
    for i in range(0,len(eigen_values) - 1):
        store = store + eigen_values[i]
        if (store / all_eig) > keep:
            return i + 1
    return len(eigen_values)

#input_data_sample()
dimension_reduced_data(x_coords, y_coords, 40, 4, 42)
#plt.savefig(sys.stdout.buffer)
#print(min_dimensions(0.9, x_coords))
#print(min_dimensions(0.95, x_coords))
#print(min_dimensions(0.98, x_coords))