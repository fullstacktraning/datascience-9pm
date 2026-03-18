import numpy as np
from sklearn.cluster import DBSCAN
X = np.array([[1,1],[1,2],[2,1],[8,8],[8,9],[50,50]])
model = DBSCAN(eps=2,min_samples=1)
labels = model.fit_predict(X)
print(labels)