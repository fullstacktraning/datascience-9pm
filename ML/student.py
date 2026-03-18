import numpy as np
from sklearn.linear_model import LogisticRegression
hours = np.array([1,2,3,4,5,6]).reshape(-1,1)
result = np.array([0,0,0,1,1,1])
model = LogisticRegression()

model.fit(hours,result)
prediction = model.predict([[3.5]])
print(prediction[0])