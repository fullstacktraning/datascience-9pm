import pandas as pd
res = pd.read_excel("students.xlsx",engine="xlrd")
print(res)

# res = pd.read_csv("students.csv")
# res["Passed"] = res["Marks"] > 80
# print(res)

# res.drop(["Passed","Name"],axis=1,inplace=True)
# print(res)

# res.drop([1,3],inplace=True)
# print(res)
# print(len(res))



#print(res[(res["Marks"]>80) & (res["Age"]>21)])
#print(res[res["Marks"]>80])
#print(res)
#print(res["RollNo"])
#print(res[["Name","Marks"]])
# print(res.loc[0])
# print("--------------------")
# print(res.iloc[0])


# data = {
#     "Name" : ["Emp1","Emp2","Emp3","Emp4","Emp5","Emp6","Emp7"],
#     "Age":[20,22,24,26,28,30,32],
#     "Salary":["1lakh","2lakh","2.5lakh","3lakh","3.5lakh","4lakh","4.5lakh"]
# }
# res = pd.DataFrame(data)
#print(res)
#print(res.head())
# print(res.tail())
# print(res.shape)
# print(res.columns)
# print(res.info())


# series = pd.Series([10,20,30,40,50])
# print(series)

