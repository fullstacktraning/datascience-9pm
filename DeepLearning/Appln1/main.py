# tensorflow - main "deep learning" library
# tensorflow - engine
import tensorflow as tf
# keras, used to build models by using tensorflow
# steering - keras
from tensorflow import keras
# import numpy
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
model = keras.Sequential([
    keras.layers.Dense(3,activation='relu'),
    keras.layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss="mean_squared_error"
)
X = np.array([[1],[2],[3],[4]],dtype=float)
y = np.array([[2],[4],[6],[8]],dtype=float)
model.fit(X,y,epochs=500)
# model - 1 epochs -- bad
# model - 100 epochs -- Better
# model -- 500 epochs --- Bit Better
# model -- 1000 epochs -- Good

class InputData(BaseModel):
    value:float

@app.get("/")
def home():
    return {"message":"welcome to fast api"} 

@app.post("/predict")
def predict(input:InputData):
    res = model.predict(np.array([[input.value]]))
    print(res)
    return {
        "result" : float(res[0][0])
    }




