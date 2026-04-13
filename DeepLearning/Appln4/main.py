# import numpy
import numpy as np
# load_digits comtaines digits (0 - 9) (encoded)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt
data = load_digits()
X = data.data
y = data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = keras.Sequential([
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test))
loss,accuracy = model.evaluate(X_test,y_test)
res = model.predict(X_test[0].reshape(1,-1))
print(np.argmax(res))
plt.imshow(X_test[0].reshape(8,8),cmap='gray')
plt.title("Predicted Image")
plt.show()