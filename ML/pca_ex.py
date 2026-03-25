import numpy as np
from sklearn.decomposition import PCA
X = np.array([[4,11],
              [8,4],
              [13,5],
              [7,14]])
model = PCA(n_components=1)
X_new = model.fit_transform(X)
print(X_new)

# (2-3.5, 4-7). [-1.5,-3]. = [1.67, 3,33]
