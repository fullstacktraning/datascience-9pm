import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data = {
    "tenure" : [1,5,10,2,8,15,3,12],
    "monthly_charges":[50,60,80,45,70,90,55,85],
    "contract-type":["Month-to-month","One year","Two year","Month-to-month","One year","Two year","Month-to-month","Two year"],
    "churn":["Yes","No","No","Yes","No","No","Yes","No"]
}
df = pd.DataFrame(data)
le = LabelEncoder()
df["contract-type"] = le.fit_transform(df["contract-type"])
df["churn"] = le.fit_transform(df["churn"])
X = df.drop("churn",axis=1)
y = df["churn"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
#print("Accuracy :",accuracy_score(y_test,y_pred))
#print("Confusion Matrix :",confusion_matrix(y_test,y_pred))
print("Classification Report :",classification_report(y_test,y_pred))
#sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
#plt.title("confusion_matrix")
#plt.show()


# new_customer = np.array([[4,65,0]])
# prediction = model.predict(new_customer)
# if prediction[0] == 1:
#     print("Churn")
# else:
#     print("No")






