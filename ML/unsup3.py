import numpy as np
from sklearn.decomposition import PCA
X = np.array([[2,4,6],
              [3,6,9],
              [4,8,12],
              [5,10,15]])
model = PCA(n_components=2)
X_new = model.fit_transform(X)
print(X_new)

# (2-3.5, 4-7). [-1.5,-3]. = [1.67, 3,33]