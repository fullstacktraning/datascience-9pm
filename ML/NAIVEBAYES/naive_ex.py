from sklearn.naive_bayes import GaussianNB
X = [[2],[5],[6],[1]]
y = ["Fail","Pass","Pass","Fail"]
model = GaussianNB()
model.fit(X,y)
res = model.predict([[4]])
print(res)

