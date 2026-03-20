from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import LinearRegression from scikit-learn module
from sklearn.linear_model import LinearRegression
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


size = np.array([500,800,1000,1200,1500]).reshape(-1,1)
price = np.array([100000,150000,200000,240000,300000])
model = LinearRegression()
model.fit(size,price)
@app.get("/predict/{sqft}")
def predict_price(sqft:int):
    new_size = np.array([[sqft]])
    predicted_price = model.predict(new_size)
   
    return {
        "Size" : sqft,
        "Predicted_Price":float(predicted_price[0]) 
    }

#print(predicted_price[0])
# plt.scatter(size,price,color="blue")
# plt.plot(size,model.predict(size),color="red")
# plt.xlabel("SQFT")
# plt.ylabel("COST")
# plt.title("FIRST AI EXAMPLE")
# plt.show()