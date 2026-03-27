import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
X = np.array([[25,40000],
     [35,60000],
     [45,80000],
     [20,20000],
     [35,120000]])
y = np.array([0,1,1,0,1])
model = DecisionTreeClassifier()
model.fit(X,y)
prediction = model.predict([[35,50000]])
print(prediction)

# plt.figure(figsize=(10,6))
# tree.plot_tree(model,filled=True)
# plt.show()