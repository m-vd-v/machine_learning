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

def set_labels():
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

labelled_data = set_labels()
print(labelled_data)



def distance(a, b) -> float:
    a_ = np.array([a[0], a[1]])
    b_ = np.array([b[0], b[1]])
    return np.sqrt(np.sum(np.square(a_ - b_)))


def linear_vector_quantization(k: int, learning_rate: float, max_epoch: int):
    prototypes = []
    for i in range(k):
        data1, data2 = split()
        point1 = data1[ np.random.randint(0, len(data1)) ]
        point2 = data2[ np.random.randint(0, len(data2)) ]
        prototypes.append([point1[0], point1[1], point1[2]])
        prototypes.append([point2[0], point2[1], point2[2]])
    prototypes = np.array(prototypes)
    print("###prototypes:", prototypes)

    prototype_trace = np.array([prototypes])
    prototype_trace = np.append(prototype_trace, [np.copy(prototypes)], axis=0)    ## making sure initial positions are accounted for
    #print("initial prototype_trace:", prototype_trace)

    Num_misclassification = []

    for i in range(max_epoch):
        #np.random.shuffle(labelled_data)
        j = 0
        for point in labelled_data:
            closest_prototype = 0
            closest_prototype_distance = distance(point, prototypes[0])
            for i in range(k*2):
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
        prototype_trace = np.append(prototype_trace, [prototypes], axis = 0)
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
        Num_misclassification.append(current_num_misclassification)

    for i in range(K):
        point = labelled_data[i]
        closest_prototype_distance = distance(point, prototypes[0])
        closest_prototype = prototypes[0]
        for prototype in prototypes:
            current_distance = distance(prototype, point)
            if closest_prototype_distance > current_distance:
                closest_prototype_distance = current_distance
                closest_prototype = prototype
        labelled_data[i][2] = closest_prototype[2]
        print(i)
    print(labelled_data)



    prototype_trace = np.array(prototype_trace)

    predicetd_labels = labelled_data[:, 2]

    return prototype_trace, predicetd_labels, Num_misclassification


def plot_lvq(k: int, learning_rate: float, max_epoch: int):
    prototype_trace, predicetd_labels, Num_misclassification = linear_vector_quantization(k=k, learning_rate=learning_rate, max_epoch=max_epoch)
    colors = ['red', 'blue', 'yellow', 'green']
    prototype_trace = np.array(prototype_trace)
    fig, ax = plt.subplots()

    data1, data2 = split()


    for i in range(2*k):
        prototype = prototype_trace[:, i]
        ax.scatter(prototype[:, 0], prototype[:, 1], edgecolors='face', c=colors[ int(prototype[0][2]) ], marker="*", s=200)
        ax.plot(prototype[:, 0], prototype[:, 1], c=colors[ int(prototype[0][2]) ])

    if len(data1) > 0:
        plt.scatter(data1[:, 0], data1[:, 1], color='red')
    if len(data2) > 0:
        plt.scatter(data2[:, 0], data2[:, 1], color='blue')
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
    #plt.savefig(sys.stdout.buffer)
    plt.show()
    plt.close()


def split():
    arr1: np.array = np.array([])
    arr2: np.array = np.array([])
    for point in labelled_data:
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

#plot_lvq(k=2, learning_rate=0.002, max_epoch=100)
#plot_error_rate(num_prototypes=2, learning_rate=0.002, max_epoch=100)
