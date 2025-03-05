import sys

import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv("../opdr3/data_clustering.csv", header=None)
data = np.array(df)


# print(np.append(x, [[3, 4]], axis=0))

## -2: VISITED
## -1: NOISE
## 0: UNVISITED
## >0: cluster id

def distance(a, b) -> float:
    return np.sqrt(np.sum(np.square(a - b)))

def add_info(D):
    i = 1
    B = np.array([[D[0, 0], D[0, 1], 0, 0]])
    while i < len(D):
        point = [D[i, 0], D[i, 1], 0, i]
        B = np.append(B, [point], axis=0)
        i += 1
    return B


def DBSCAN(D, eps, MinPts):
    cluster = 0
    D = add_info(D)
    i = 0
    while i < len(D):
        if D[i, 2] == 0:
            point = D[i]
            D[i, 2] = -2
            neighbor_pts = region_query(D, point, eps)
            # print("lenght neighbor_pts", len(neighbor_pts))
            if len(neighbor_pts) < MinPts:
                D[i, 2] = -1
            else:
                cluster += 1
                expand_cluster(D, point, neighbor_pts, cluster, eps, MinPts)
        i += 1
    labels = D[:, 2].astype(int)
    print("final", labels)
    return labels


def region_query(D, P, eps):
    N = np.array([])
    for other_point in D:
        if distance(P[:2], other_point[:2]) <= eps:
            # print("otherpoint, N:", other_point, N)
            if len(N) == 0:
                N = [other_point]
            else:
                N = np.append(N, [other_point], axis=0)
    # print("N: ", N)
    return N


def expand_cluster(D, P, neighbor_points, cluster, eps, MinPts):
    P[2] = cluster
    # print("cluser", cluster)
    # print("neigbors", neighbor_points)
    # print("nb points: ", neighbor_points)
    i = 0
    while i < len(neighbor_points):
        point = neighbor_points[i]
        if point[2] <= 0:
            point[2] = -2
            other_neighbor_points = region_query(D, point, eps)
            # print("neighbours of neighbor",len(other_neighbor_points))
            if len(other_neighbor_points) >= MinPts:
                neighbor_points = np.append(neighbor_points, other_neighbor_points, axis=0)
        if point[2] <= 0:  ## not part of any cluster
            point[2] = cluster
            D[(int)(point[3]), 2] = cluster
        i += 1


def plot_db_scan(D, eps, k):
    cluster_labels = DBSCAN(D, eps, k)
    plt.figure()
    scatter = plt.scatter(D[:, 0], D[:, 1], c=cluster_labels, edgecolors="k")
    plt.title(f"DBSCAN clustering with MinPt={k},eps={eps}")
    plt.xlabel('First feature')
    plt.ylabel('Second feature')

    legend = plt.legend(*scatter.legend_elements(num=sorted(np.unique(cluster_labels))), title="Clusters")
    plt.gca().add_artist(legend)
    plt.savefig(sys.stdout.buffer)
    plt.show()
    plt.close()


###################################################

class Neighbor:
    index: int
    distance: float

    def __init__(self, index: int, distance: float):
        self.index = index
        self.distance = distance

def neighbor_seen(neighbors: [Neighbor], current_index: int) -> bool:
    for neighbor in neighbors:
        if neighbor.index == current_index:
            return True
    return False


def get_furthest_neighbor(neighbors: [Neighbor]) -> Neighbor:
    furthest_neighbor = neighbors[0]
    for neighbor in neighbors:
        if neighbor.distance > furthest_neighbor.distance:
            furthest_neighbor = neighbor
    return furthest_neighbor


## replaces the furthest away neighbor with the current_neighbor, if the current_neighbor is closer than the furthest.
def replace_furthest_neighbor(nearest_neighbors: [Neighbor], current_neighbor: Neighbor) -> [Neighbor]:
    furthest_neighbor: Neighbor = get_furthest_neighbor(nearest_neighbors)
    if current_neighbor.distance < furthest_neighbor.distance:
        nearest_neighbors.remove(furthest_neighbor)
        nearest_neighbors.append(current_neighbor)
    return nearest_neighbors


def avg_distances(neighbors: [Neighbor]) -> float:
    print(neighbors)
    sum: float = 0.0
    for neighbor in neighbors:
        sum += neighbor.distance
    return sum  # / (len(neighbors)-1)


def plot_knn(D, k, y):
    xpoints = np.array([])
    ypoints = np.array([])

    for i in range(len(D)):
        xpoints = np.append(xpoints, i)

    for i, point in enumerate(D):
        nearest_neighbors: [Neighbor] = []
        # fill nearest_neighbors with infinitely far away neighbors
        for _i in range(k):
            nearest_neighbors.append( Neighbor(-1, float('inf')) )

        for j, neighbor_point in enumerate(D):
            if neighbor_seen(nearest_neighbors, j):
                continue
            neighbor = Neighbor(j, distance(D[i], D[j]))
            replace_furthest_neighbor(nearest_neighbors, neighbor)

        avg_distance = avg_distances(nearest_neighbors)
        ypoints = np.append(ypoints, avg_distance)

    ypoints = np.sort(ypoints)

    plt.plot(xpoints, ypoints)
    plt.title(str(k) + ' Nearest Neighbor graph')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(str(k) + '-Nearest Neighbor Distance')
    if y:
        plt.axhline(y, linestyle='--')
    plt.show()
    pass


# 0
# DBSCAN(data, 0.04, 3)
# y should be an optional parameter, setting it to None should do nothing.. Setting it to anything else (int) should draw a line plt.axhline(y, linestyle='--')
# 1
plot_knn(D=data, k=2, y=None)
# 2
plot_knn(D=data, k=3, y=None)
# 3
plot_knn(D=data, k=4, y=None)
# 4
# plot_db_scan(D=data, eps=0.04, k=2,)
# 5
# plot_db_scan(D=data, eps=0.04, k=3)
# 6
# plot_db_scan(D=data, eps=0.04, k=4,)
