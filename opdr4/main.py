import random
import sys

import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv("simplevqdata.csv", header=None)
data = np.array(df)
#print(data)

N: int = len(data[0])   # 2
P: int = len(data)      # 1000

#print(N, P)

n: float = 0.10     # learning rate of 10%

np.random.seed(1740844038)
np.set_printoptions(precision=5, suppress=True)

def distance(a, b) -> float:
    return np.sqrt(np.sum(np.square(a - b)))


def vector_quantization(k: int, learning_rate: float, max_epoch: int, stupid_init: bool = False):
    prototypes = []
    for i in range(k):
        point = random.choice(data)
        prototypes.append([point[0], point[1]])
    prototypes = np.array(prototypes)
    #print("prototypes:", prototypes)

    ## Setting the position of each prototype to x=0.0, y=0.0 for "stupid" initialisation.
    if stupid_init:
        for prototype in prototypes:
            prototype[0] = 10
            prototype[1] = 10

    HVQ_trace = []

    prototype_trace = np.array([prototypes])
    prototype_trace = np.append(prototype_trace, [np.copy(prototypes)], axis=0)    ## making sure initial positions are accounted for
    #print("initial prototype_trace:", prototype_trace)

    for i in range(max_epoch):
        np.random.shuffle(data)
        j = 0
        for point in data:
            closest_prototype = 0
            closest_prototype_distance = distance(point, prototypes[0])
            for i in range(k):
                current_distance = distance(prototypes[i], point)
                if closest_prototype_distance > current_distance:
                    closest_prototype = i
                    closest_prototype_distance = current_distance
            prototypes[closest_prototype][0] += (point[0] - prototypes[closest_prototype][0]) * learning_rate
            prototypes[closest_prototype][1] += (point[1] - prototypes[closest_prototype][1]) * learning_rate
            #print(j, " :new prototypes:", prototypes)
            j = j + 1
        prototype_trace = np.append(prototype_trace, [prototypes], axis = 0)
        #print("added prototype:", prototypes)

        distance_sum = 0.0
        for point in data:
            closest_prototype_distance = distance(point, prototypes[0])
            for prototype in prototypes:
                current_distance = distance(prototype, point)
                if closest_prototype_distance > current_distance:
                    closest_prototype_distance = current_distance
            distance_sum += closest_prototype_distance
        HVQ_trace.append(distance_sum)

    #print("HVQ_trace (length=", len(HVQ_trace), "): ", HVQ_trace)
    #print("prototype_trace: ", prototype_trace)

    prototype_trace = np.array(prototype_trace)

    return (prototype_trace, HVQ_trace)


def plot_vq(k: int, learning_rate: float, max_epoch: int, stupid_init: bool = False):
    prototype_trace, HVQ_trace = vector_quantization(k=k, learning_rate=learning_rate,
                                                     max_epoch=max_epoch, stupid_init=stupid_init)
    colors = ['red', 'blue', 'yellow', 'green']
    prototype_trace = np.array(prototype_trace)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], edgecolors='k')

    for i in range(k):
        prototype = prototype_trace[:, i]
        #print("prototype col:", prototype)
        ax.scatter(prototype[:, 0], prototype[:, 1], edgecolors='face', c=colors[i])
        ax.plot(prototype[:, 0], prototype[:, 1], c=colors[i])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    plt.savefig(sys.stdout.buffer)
    plt.show()
    plt.close()


def plot_vq_error(HVQerror_k, max_epoch: int):
    fig, ax = plt.subplots()
    #print("errors", HVQerror_k)
    ax.plot(range(max_epoch), HVQerror_k)
    ax.set(xlabel="Epoch", ylabel="Quantization error", title="Quantization Error Over Epochs")
    plt.show()
    plt.savefig(sys.stdout.buffer)


#_, errors = vector_quantization(2, 0.01, 100)
#plot_vq(k=2, learning_rate=0.001, max_epoch=100)
#plot_vq(k=4, learning_rate=0.001, max_epoch=100)
#plot_vq_error(HVQerror_k=errors, max_epoch=100)


def plot_vq_and_error(k, learning_rate, max_epoch):
    print(f"Testing k={k}, learning_rate={learning_rate}, max_epoch={max_epoch}")
    #plot_vq(k, learning_rate, max_epoch)
    _, errors = vector_quantization(k, learning_rate, max_epoch)
    plot_vq_error(HVQerror_k=errors, max_epoch=max_epoch)


test1 = False
if test1:
    max_epoch = 50
    plot_vq_and_error(k=2, learning_rate=0.1, max_epoch=max_epoch)
    plot_vq_and_error(k=2, learning_rate=0.005, max_epoch=max_epoch)
    plot_vq_and_error(k=2, learning_rate=0.001, max_epoch=max_epoch)
    plot_vq_and_error(k=4, learning_rate=0.1, max_epoch=max_epoch)
    plot_vq_and_error(k=4, learning_rate=0.005, max_epoch=max_epoch)
    plot_vq_and_error(k=4, learning_rate=0.001, max_epoch=max_epoch)

test2 = False
if test2:
    plot_vq(k=2, learning_rate=0.001, max_epoch=100, stupid_init=True)
    plot_vq(k=4, learning_rate=0.001, max_epoch=100, stupid_init=True)

inputs = True
if inputs:
    ## 1
    prototype_trace, HVQ_trace = vector_quantization(
        k=2,
        learning_rate=0.1,
        max_epoch=100,
    )
    print(prototype_trace)
    print(np.round(HVQ_trace, decimals=5))

    ## 2
    prototype_trace, HVQ_trace = vector_quantization(
        k=4,
        learning_rate=0.1,
        max_epoch=100,
    )
    print(prototype_trace)
    print(np.round(HVQ_trace, decimals=5))

    ## 3
    plot_vq(
        k=2,
        learning_rate=0.1,
        max_epoch=100,
    )

    ## 4
    max_epoch = 100
    prototypes_k, HVQerror_k = vector_quantization(
        k=2,
        learning_rate=0.1,
        max_epoch=max_epoch,
    )
    plot_vq_error(
        HVQerror_k=HVQerror_k,
        max_epoch=max_epoch,
    )