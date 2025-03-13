import random
import sys

import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv("data_lvq.csv", header=None)
data = np.array(df)

K = len(data)
t_max = 100

np.random.seed(1740844038)
np.set_printoptions(
    precision=5,
    suppress=True,
    threshold=sys.maxsize
)

def set_labels(data):
    first_point = np.array([data[0][0], data[0][1], 0])
    new_data = np.array([first_point])
    i = 1
    while i < K:
        label = 0
        if i >= K/2:
            label = 1
        new_point = np.array([ data[i][0], data[i][1], label ])
        new_data = np.append(new_data, [new_point], axis=0)
        i += 1
    return new_data
#print(labelled_data)



def distance(a, b) -> float:
    a_ = np.array([a[0], a[1]])
    b_ = np.array([b[0], b[1]])
    return np.sqrt(np.sum(np.square(a_ - b_)))


def linear_vector_quantization(num_prototypes: int, learning_rate: float, max_epoch: int):
    labelled_data = set_labels(data)
    prototypes = []
    data1, data2 = split(labelled_data)
    for i in range(num_prototypes):
        point1 = data1[ np.random.randint(0, len(data1)) ]
        point2 = data2[ np.random.randint(0, len(data2)) ]
        prototypes.append([point1[0], point1[1], point1[2]])
        prototypes.append([point2[0], point2[1], point2[2]])
    prototypes = np.array(prototypes)
    #print("###prototypes:", prototypes)

    print("prototypes: ", prototypes)
    prototype_trace = np.array([prototypes[:, :2]])
    #print("initial prototype_trace:", prototype_trace)

    num_misclassification = []

    for i in range(max_epoch):
        #np.random.shuffle(labelled_data)
        j = 0
        for point in labelled_data:
            closest_prototype = 0
            closest_prototype_distance = distance(point, prototypes[0])
            for i in range(num_prototypes*2):
                current_distance = distance(prototypes[i], point)
                if closest_prototype_distance > current_distance:
                    closest_prototype = i
                    closest_prototype_distance = current_distance

            if point[2] == prototypes[closest_prototype][2]:   ## is part same class
                prototypes[closest_prototype][0] += (point[0] - prototypes[closest_prototype][0]) * learning_rate
                prototypes[closest_prototype][1] += (point[1] - prototypes[closest_prototype][1]) * learning_rate
            else:
                prototypes[closest_prototype][0] -= (point[0] - prototypes[closest_prototype][0]) * learning_rate
                prototypes[closest_prototype][1] -= (point[1] - prototypes[closest_prototype][1]) * learning_rate
            #print(j, " :new prototypes:", prototypes)
            j = j + 1
        prototype_trace = np.append(prototype_trace, [prototypes[:, :2]], axis=0)
        #print("added prototype:", prototypes)

        current_num_misclassification = 0
        for point in labelled_data:
            closest_prototype_distance = distance(point, prototypes[0])
            closest_prototype = prototypes[0]
            for prototype in prototypes:
                current_distance = distance(prototype, point)
                if closest_prototype_distance > current_distance:
                    closest_prototype_distance = current_distance
                    closest_prototype = prototype

            if point[2] != closest_prototype[2]:
                current_num_misclassification += 1
        num_misclassification.append(current_num_misclassification)

    labels = []
    for i in range(K):
        point = data[i]
        closest_prototype_distance = distance(point, prototypes[0])
        closest_prototype = prototypes[0]
        for prototype in prototypes:
            current_distance = distance(prototype, point)
            if closest_prototype_distance > current_distance:
                closest_prototype_distance = current_distance
                closest_prototype = prototype
        labels.append(int(closest_prototype[2]))
    #print(labels)

    prototype_trace = np.array(prototype_trace)

    return prototype_trace, labels, num_misclassification


def plot_trajectory(num_prototypes: int, learning_rate: float, max_epoch: int):
    prototype_trace, predicted_labels, num_misclassification = linear_vector_quantization(num_prototypes=num_prototypes,
                                                                                          learning_rate=learning_rate,
                                                                                          max_epoch=max_epoch)

    print(prototype_trace)

    colors = ['red', 'blue']
    prototype_trace = np.array(prototype_trace)
    fig, ax = plt.subplots()

    for i in range(K):
        plt.scatter(data[i, 0], data[i, 1], color=colors[predicted_labels[i]])
    for i in range(2 * num_prototypes):
        prototype = prototype_trace[:, i]
        ax.scatter(prototype[:, 0], prototype[:, 1], edgecolors='face', c=colors[i%2], marker="*",
                   s=200)
        ax.plot(prototype[:, 0], prototype[:, 1], c=colors[i%2])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.title('Trajectory Of Prototypes')
    # plt.savefig(sys.stdout.buffer)
    plt.show()
    plt.close()


def plot_error_rate(num_prototypes, learning_rate, max_epoch: int):
    _, _, error_rate = linear_vector_quantization(num_prototypes, learning_rate, max_epoch)

    for i in range(len(error_rate)):
        error_rate[i] = error_rate[i]/100

    fig, ax = plt.subplots()
    #print("errors", HVQerror_k)
    ax.plot(range(max_epoch), error_rate)

    plt.xlabel('Epoch')
    plt.ylabel('The error rate in %')
    plt.title('Learning curve')
    plt.savefig(sys.stdout.buffer)
    #plt.show()
    plt.close()


def split(dataset):
    arr1: np.array = np.array([])
    arr2: np.array = np.array([])
    for point in dataset:
        if point[2] == 0:
            if len(arr1) <= 0:
                arr1 = np.array([point])
            else:
                arr1 = np.append(arr1, [point], axis=0)
        else:
            if len(arr2) <= 0:
                arr2 = np.array([point])
            else:
                arr2 = np.append(arr2, [point], axis=0)
    return arr1, arr2

def plot_data():
    labelled_data = set_labels(data)
    data1, data2 = split(labelled_data)

    plt.scatter(data1[:, 0], data1[:, 1], color='red')
    plt.scatter(data2[:, 0], data2[:, 1], color='blue')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    plt.savefig(sys.stdout.buffer)
    plt.close()

#plot_data()
#plot_trajectory(num_prototypes=2, learning_rate=0.002, max_epoch=100)
#plot_error_rate(num_prototypes=2, learning_rate=0.002, max_epoch=100)
