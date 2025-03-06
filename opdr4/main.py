import random
import sys

import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

df = pandas.read_csv("simplevqdata.csv", header=None)
data = np.array(df)
print(data)

N: int = len(data[0])   # 2
P: int = len(data)      # 1000

print(N, P)

n: float = 0.10     # learning rate of 10%

def distance(a, b) -> float:
    return np.sqrt(np.sum(np.square(a - b)))


def vector_quantization(k: int, learning_rate: float, max_epoch: int):
    prototypes = []
    for i in range(k):
        prototypes.append(data[ random.randrange(0, P) ])
    print("prototypes:", prototypes)

    HVQ_trace = []

    prototype_trace = []
    for i in range(max_epoch):
        distance_sum = 0.0
        for point in data:
            closest_prototype = prototypes[0]
            closest_prototype_distance = distance(point, prototypes[0])
            for prototype in prototypes:
                current_distance = distance(prototype, point)
                if closest_prototype_distance > current_distance:
                    closest_prototype = prototype
                    closest_prototype_distance = current_distance
            closest_prototype[0] += (point[0] - closest_prototype[0]) * learning_rate
            closest_prototype[1] += (point[1] - closest_prototype[1]) * learning_rate
        prototype_trace.append(prototypes)

        for point in data:
            closest_prototype = prototypes[0]
            closest_prototype_distance = distance(point, prototypes[0])
            for prototype in prototypes:
                current_distance = distance(prototype, point)
                if closest_prototype_distance > current_distance:
                    closest_prototype = prototype
                    closest_prototype_distance = current_distance
            distance_sum += closest_prototype_distance
        HVQ_trace.append(distance_sum)

    print("HVQ_trace (length=", len(HVQ_trace), "): ", HVQ_trace)
    print("prototype_trace: ", prototype_trace)

    return (prototype_trace, HVQ_trace)


def plot_vq(k: int, learning_rate: float, max_epoch: int):
    prototype_trace, HVQ_trace = vector_quantization(k, learning_rate, max_epoch)
    colors = ['red', 'blue', 'yellow', 'green']

    fig, ax = plt.subplots()
    ax.scatter(data[0], data[1], edgecolors='k')

    for i in range(k):
        protoype = prototype_trace[i]
        ax.scatter(protoype[0], protoype[1], edgecolors='k', c=colors[i])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    #plt.savefig(sys.stdout.buffer)
    plt.show()
    plt.close()


def plot_vq_error(HVQerror_k, max_epoch: int):
    pass


#vector_quantization(2, 0.1, 50)
plot_vq(k=2, learning_rate=0.1, max_epoch=100)