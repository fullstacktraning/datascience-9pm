import numpy as np
import matplotlib.pyplot as plt
# KMeans Alogorithm from scikit-learn
from sklearn.cluster import KMeans

X = np.array([[1,2],[1,4],[1,0],
              [10,2],[10,4],[10,0],
              [20,7],[20,3],[20,1]]) #3 clusters


kmeans = KMeans(n_clusters=3)
# Cluster - 1 [1,2],[1,4],[1,0]
# Cluster - 2 [10,2],[10,4],[10,0]

kmeans.fit(X)
#print(X)
print(X[:,0])
print(X[:,1])
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_)
plt.show()