import pandas

import numpy as np

df = pandas.read_csv("../opdr3/data_clustering.csv", header=None)
x = np.array(df)

#print(np.append(x, [[3, 4]], axis=0))

## -2: VISITED
## -1: NOISE
## 0: UNVISITED
## >0: cluster id

def distance(a, b) -> float:
    return np.linalg.norm(a - b)

def DBSCAN(D, eps, MinPts):
    cluster = 0
    D = np.pad(D, ((0, 0), (0, 1)))
    i = 0
    while i < len(D):
        point = D[i]
        D[i, 2] = -2
        neighbor_pts = regionQuery(D, point, eps)
        print("lenght neighbor_pts", len(neighbor_pts))
        if len(neighbor_pts) < MinPts:
            D[i, 2] = -1
        else:
            cluster += 1
            expandCluster(D, point, neighbor_pts, cluster, eps, MinPts)
        i += 1
    print(D)


def regionQuery(D, P, eps):
    N = np.array([])
    for other_point in D:
        if distance(P[:2], other_point[:2]) <= eps:
            #print("otherpoint, N:", other_point, N)
            if len(N) == 0:
                N = [other_point]
            else:
                N = np.append(N, [other_point], axis=0)
    #print("N: ", N)
    return N


def expandCluster(D, P, neighbor_points, cluster, eps, MinPts):
    P[2] = cluster
    print("cluser", cluster)
    #print("nb points: ", neighbor_points)
    for other_point in neighbor_points:
        if other_point[2] != 0:
            other_point[2] = -2
            #other_neighbor_points = regionQuery(D, other_point, eps)
            #if len(other_neighbor_points) >= MinPts:
                #neighbor_points = np.append(neighbor_points, [other_point], axis=0)
        if other_point[2] <= 0: ## not part of any cluster
            other_point[2] = cluster


DBSCAN(x, 0.04, 3)