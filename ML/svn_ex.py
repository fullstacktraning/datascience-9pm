from sklearn import svm
import matplotlib.pyplot as plt
# excel sheet
X = [[1],[2],[4],[5]]
y = [0,0,1,1]
model = svm.SVC(kernel="linear")
model.fit(X,y)
print(model.coef_)
print(model.intercept_)
# swagger
print( model.predict([[10]]) )
plt.scatter(X,y,color="red")
plt.show()


# wx + b = 1x10 - 3 = 7 (POSITIVE)
# 3.5 = 1x3.5 - 3 = 0.5 (POSITIVE)
# 1.5 = 1x1.5 - 3 = -1.5 (NEGATIVE)
# boundry = closest negative support vector + closest positive support vector / 2
# x - 3 = 0
# 1w -3 = 0
# x = 1  b = -3