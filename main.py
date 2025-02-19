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

# output is a tuple of 3 arrays
def pca(x: numpy.ndarray[numpy.ndarray], d: int, debug: bool = False) -> tuple:
    x = x.T  # transposing the array, so we can easily iterate over columns instead of rows
    for i, column in enumerate(x):
        mean = column.mean()
        st_dev = column.std()
        x[i] = (x[i] - mean)/st_dev
        if debug:
            print("standardized mean and st_dev of column", i, "=", x[i].mean(), ",", x[i].std())
            ## The mean and st_dev of each column are 0 and 1 respectively (accounting for rounding errors)
    Z = x.T    # transposing again to original state to get standardized array data set Z
    if debug:
        print("Z: ", Z)
    covar = np.cov(Z, rowvar=False)
    eigen_values: numpy.ndarray
    eigen_vectors: numpy.ndarray
    eigen_values, eigen_vectors = numpy.linalg.eig(covar)
    if debug:
        print("eigen values: ", eigen_values)
    # sorting the eigen_values and moving the eigen_vectors to their correct positions based on the sort
    # according to https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
    idx = eigen_values.argsort()[::-1]  # sorting and then reversing array to get an array sorted from high to low
    eigen_values = eigen_values[idx]  # sorted eigenvalues are correct when comparing with output in themis
    eigen_vectors = eigen_vectors[:, idx]
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

print(pca(x_coords, 40, debug=False))

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
    fig, ax = plt.subplots()
    ax.plot(eigen_values,"purple")
    ax.set(xlabel="Index eigen-value", ylabel="Eigen-value", title="Eigen-value Profile of the Dataset")
    plt.show()

eigen_value_profile(x_coords, 40)


########## [ INPUT 4 ] #################################################################################

def dimension_reduced_data(x: numpy.array(numpy.array(float)), y: numpy.array(float),
                           d: int, perplexity: int, random_state: int):
    _, _, x_reduced = pca(x, d)
    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    x_tsne = tsne.fit_transform(x)
    # plot
    fig, ax = plt.subplots()

    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:,1], c=y)
    labels = ['object{}'.format(i) for i in range(1, 20)]
    ax.set(xlim=(-100, 100), xticks=np.arange(-100, 125, 25),
           ylim=(-100, 100), yticks=np.arange(-100, 100, 25),
           xlabel="t-sne 1", ylabel="t-sne 2", title="t-sne visualisation of dimension reduced data")
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(*scatter.legend_elements(), title="Object ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(sys.stdout.buffer)
    plt.show()

    fig, ax = plt.subplots()
    for i in range(1,20):
        ax.scatter([x_tsne[i, 0]], [x_tsne[i,1]], label="class " + str(i))
        i = i + 1
    ax.set(xlabel="t-sne 1", ylabel="t-sne 2", title="t-sne visualisation of dimension reduced data")
    plt.legend(title="Object ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()



#input_data_sample()
#dimension_reduced_data(x_coords, y_coords, 40, 4, 42)
#plt.savefig(sys.stdout.buffer)