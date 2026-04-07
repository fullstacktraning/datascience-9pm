import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[150,50],[160,60],[170,80]])

y = np.array(["Slim","Slim","Fat"])

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X,y)

new_student = np.array([[1,3]])

prediction = model.predict([[165,65]])

print(prediction)