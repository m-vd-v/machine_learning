
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

df = pandas.read_csv("data_clustering.csv", header=None)
x = np.array(df)
print(x)
#### [INPUT -1] ####

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
    plt.scatter(x[:, 0], x[:, 1], c=labels, edgecolors="k")
    plt.title(f"Clustering results for {k} clusters, using '{measure}' measure")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()


# This test case will not be graded, it is simply for you to check whether your data matches
# You will always pass this test case, you will have to diff the image yourself.
# -1 ungraded
plot_data_using_scatter_plot()

# 0
# !! plot me with plt.xticks([])  to remove x-axis labels
plot_dendrogram(linkage_measure = "single", calc_thresholds = False)

# 1
plot_dendrogram(linkage_measure = "average", calc_thresholds = True)

# 2
plot_dendrogram(linkage_measure = "complete", calc_thresholds = False)

# 3
plot_dendrogram(linkage_measure = "ward", calc_thresholds = False)

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
agglomerative_clustering(measure = "single", k = 2)

# 5
agglomerative_clustering(measure = "average", k = 3)

# 6
agglomerative_clustering(measure = "complete", k = 4)

# 7
agglomerative_clustering(measure = "single", k = 3)

# 8
agglomerative_clustering(measure = "single", k = 4)

# 9
agglomerative_clustering(measure = "average", k = 4)