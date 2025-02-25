import numpy
import pandas
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy

df = pandas.read_csv("data_clustering.csv", header=None)
x = np.array(df)


#### [INPUT -1] ####

def distance(a, b) -> float:
    ax = a[0]
    ay = a[1]
    bx = b[0]
    by = b[1]
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return numpy.sqrt(pow(dx, 2) + pow(dy, 2))


def calc_silhouette_score(labels: numpy.ndarray):
    cluster_amt: int = labels.max() + 1
    clusters = []
    for i in range(cluster_amt):
        clusters.append([])
    for i in range(labels.size):
        label = labels[i]
        clusters[label].append(x[i])

    for i, cluster in enumerate(clusters):
        print(f"cluster({i}): len of {len(cluster)}")

    a_values = []
    b_values = []
    for cluster_i in clusters:
        sum_: float = 0.0  # declared as sum_ so it doesn't shadow built-in sum
        for i in cluster_i:
            for j in cluster_i:
                sum_ += distance(i, j)
        a_i = (1 / (len(cluster_i) - 1)) * sum_
        a_values.append(a_i)

        min_distance: float = None
        for cluster_k in clusters:
            if cluster_i == cluster_k:
                continue
            for i in cluster_i:
                for j in cluster_k:
                    current_distance = distance(i, j)
                    if min_distance is None or current_distance < min_distance:
                        min_distance = current_distance
        b_values.append(min_distance)
    ##### end of loop
    sum_ = 0
    for i in range(cluster_amt):
        a_i = a_values[i]
        b_i = b_values[i]
        sum_ += ( (b_i - a_i) / (max(a_i, b_i)) )
    S = (1/cluster_amt) * sum_

    print("real silhouette score:", silhouette_score(x, labels))

    return S



def plot_data_using_scatter_plot():
    plt.scatter(x[:, 0], x[:, 1])
    plt.title("Scatter plot - original data")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()


#### [INPUT -1] ####

def plot_dendrogram(linkage_measure: str, calc_thresholds: bool):
    dend = hierarchy.linkage(x, method=linkage_measure)
    hierarchy.dendrogram(dend, no_labels=True)
    plt.title(f"Dendogram - {linkage_measure} measure")
    plt.xlabel("Observations")
    plt.ylabel("Dissimilarity")
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


plot_dendrogram(linkage_measure="single", calc_thresholds=True)

# This test case will not be graded, it is simply for you to check whether your data matches
# You will always pass this test case, you will have to diff the image yourself.
# -1 ungraded
plot_data_using_scatter_plot()

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
'''
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
