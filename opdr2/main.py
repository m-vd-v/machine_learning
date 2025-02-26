import numpy
import pandas
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
import scipy

df = pandas.read_csv("data_clustering.csv", header=None)
x = np.array(df)

def distance(a, b) -> float:
    return numpy.linalg.norm(a - b)

def get_points_in_cluster(labels, points, cluster_id) -> list:
    points_in_cluster = []
    for i, label in enumerate(labels):
        if label != cluster_id:
            continue
        point = points[i]
        points_in_cluster.append(point)
    return points_in_cluster

def get_points_outside_cluster(labels, points, cluster_id) -> list:
    points_outside_cluster: Array = []
    for i, label in enumerate(labels):
        if label == cluster_id:
            continue
        point = points[i]
        points_outside_cluster.append(point)
    return points_outside_cluster

def calc_silhouette_score(labels: numpy.ndarray):
    cluster_amt = labels.max() + 1

    S_scores = []

    for i in range(len(x)):
        current_point = x[i]
        cluster_i = labels[i]
        points_in_cluster_i = get_points_in_cluster(labels, x, cluster_i)
        points_outside_cluster = get_points_outside_cluster(labels, x, cluster_i)

        sum_ = 0.0  # declaring sum_ because sum would shadow an in-built function
        for compare_point in points_in_cluster_i:
            sum_ += distance(current_point, compare_point)
        a_i = 0
        if len(points_in_cluster_i) > 1:   # make sure not to divide by 0
            a_i = sum_ / (len(points_in_cluster_i) - 1)

        b_i = float('inf')
        for cluster_k in range(cluster_amt):
            if cluster_k == cluster_i:
                continue
            points_in_cluster_k = get_points_in_cluster(labels, x, cluster_k)
            sum_ = 0
            for compare_point in points_in_cluster_k:
                sum_ += distance(current_point, compare_point)
            current_b_i = sum_ / len(points_in_cluster_k)
            if current_b_i < b_i:
                b_i = current_b_i

        s_i = 0
        if max(a_i, b_i) != 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        S_scores.append(s_i)

    S = numpy.mean(S_scores)
    print("own silhouette score: ", S)
    print("true silhouette score:", sklearn.metrics.silhouette_score(x, labels))
    return S





def plot_data_using_scatter_plot():
    plt.scatter(x[:, 0], x[:, 1])
    plt.title("Scatter plot - original data")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()


#### [INPUT -1] ####

def plot_dendrogram(linkage_measure: str, calc_thresholds: bool):
    z = hierarchy.linkage(x, method=linkage_measure)
    hierarchy.dendrogram(z)
    plt.xticks([])
    plt.title(f"Dendogram - {linkage_measure} measure")
    plt.xlabel("Observations")
    plt.ylabel("Dissimilarity")
    if calc_thresholds:
        two_threshold = (z[198, 2] + z[197, 2])/2
        three_threshold = (z[197, 2] + z[196, 2])/2
        four_threshold = (z[196, 2] + z[195, 2])/2
        plt.axhline(y=two_threshold, c='green', linestyle='dashed')
        plt.axhline(y=three_threshold, c='green', linestyle='dashed')
        plt.axhline(y=four_threshold, c='green', linestyle='dashed')

    plt.show()


def agglomerative_clustering(measure: str, k: int):
    clustering = AgglomerativeClustering(n_clusters=k, linkage=measure).fit(x)
    labels = clustering.labels_
    print(calc_silhouette_score(labels))
    plt.scatter(x[:, 0], x[:, 1], c=labels, edgecolors="k")
    plt.title(f"Clustering results for {k} clusters, using '{measure}' measure")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()


#plot_dendrogram(linkage_measure="single", calc_thresholds=True)

# This test case will not be graded, it is simply for you to check whether your data matches
# You will always pass this test case, you will have to diff the image yourself.
# -1 ungraded
#plot_data_using_scatter_plot()

# 0
# !! plot me with plt.xticks([])  to remove x-axis labels
# plot_dendrogram(linkage_measure = "single", calc_thresholds = False)

# 1
# plot_dendrogram(linkage_measure = "average", calc_thresholds = True)

# 2
# plot_dendrogram(linkage_measure = "complete", calc_thresholds = False)

# 3
# plot_dendrogram(linkage_measure = "ward", calc_thresholds = False)

# 4
'''
This is how we plot it :

plt.scatter(<for you to decide>, c=labels, edgecolors="k")
plt.title(f"Clustering results for {k} clusters, using '{measure}' measure")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.savefig(sys.stdout.buffer)
plt.close()

agglomerative_clustering(measure="single", k=2)

# 5
agglomerative_clustering(measure="average", k=3)

# 6
agglomerative_clustering(measure="complete", k=4)

# 7
agglomerative_clustering(measure="single", k=3)

# 8
agglomerative_clustering(measure="single", k=4)

# 9
agglomerative_clustering(measure="average", k=4)

'''
#results
#dendrograms
#plot_dendrogram(linkage_measure = "average", calc_thresholds = True)
#plot_dendrogram(linkage_measure = "single", calc_thresholds = True)
#plot_dendrogram(linkage_measure = "complete", calc_thresholds = True)
#plot_dendrogram(linkage_measure = "ward", calc_thresholds = True)


#agglomerative clustering
#plot_data_using_scatter_plot()

agglomerative_clustering(measure="average", k=2)
agglomerative_clustering(measure="average", k=3)
agglomerative_clustering(measure="average", k=4)

agglomerative_clustering(measure="complete", k=2)
agglomerative_clustering(measure="complete", k=3)
agglomerative_clustering(measure="complete", k=4)

agglomerative_clustering(measure="single", k=2)
agglomerative_clustering(measure="single", k=3)
agglomerative_clustering(measure="single", k=4)

agglomerative_clustering(measure="ward", k=2)
agglomerative_clustering(measure="ward", k=3)
agglomerative_clustering(measure="ward", k=4)
