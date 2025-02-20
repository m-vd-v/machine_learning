


#### [INPUT -1] ####

def plot_data_using_scatter_plot():
    pass


#### [INPUT -1] ####

def plot_dendrogram(linkage_measure: str, calc_thresholds: bool):
    match linkage_measure:
        case "single":
            pass
        case "average":
            pass
        case "complete":
            pass


def agglomerative_clustering(measure: str, k: int):
    pass






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