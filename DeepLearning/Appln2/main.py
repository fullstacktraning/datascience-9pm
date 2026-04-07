import tensorflow as tf
from tensorflow import keras
import numpy as np
X = np.array([
    [1, 85, 66, 26.6, 31],
    [8, 183, 64, 23.3, 32],
    [1, 89, 66, 28.1, 21],
    [0, 137, 40, 43.1, 33],
    [5, 116, 74, 25.6, 30]
], dtype=float)
y = np.array([
    [0],
    [1],
    [0],
    [1],
    [0]
], dtype=float)
model = keras.Sequential([
    keras.layers.Dense(8,activation='relu'),
    keras.layers.Dense(4,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
# relu --> removes negatives output : -10.8 --> 0
# sigmoid ---> Binary 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X,y,epochs=200)
res =  model.predict(np.array([[2,120,70,30.5,35]])) 
print(res)
if res[0][0] > 0.5:
    print("Diabetic")
else:
    print("Not Diabetic")


