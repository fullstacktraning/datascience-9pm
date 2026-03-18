import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
X = np.array([[1,60],
              [2,65],
              [3,70],
              [4,80],
              [5,85],
              [6,90],
              [7,95],
              [8,98]])
y = np.array([0,0,0,1,1,1,1,1])
model = RandomForestClassifier(n_estimators=10)
model.fit(X,y)
prediction = model.predict([[4,70]])
print(prediction)
acc = model.score(X,y)
print(acc)