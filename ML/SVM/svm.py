import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X = np.array([[2,2],[4,4],[2,4],[4,2],[6,6],[8,8],[6,8],[8,6]])
y = np.array([0,0,0,0,1,1,1,1])

model = svm.SVC(kernel='linear')
model.fit(X,y)
prediction = model.predict([[5,5]])

print(prediction)

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()