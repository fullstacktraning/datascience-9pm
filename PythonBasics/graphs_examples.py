import pandas as pd
import matplotlib.pyplot as plt       
df = pd.read_excel("Book1.xlsx", engine="openpyxl")
names = df["Name"]
marks = df["Marks"]
bars = plt.bar(names,marks,color="teal")
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        bar.get_height(),
        ha="center",
        va="bottom"
    )

plt.title("Student Marks")
plt.ylabel("Marks")
plt.show()





# import matplotlib.pyplot as plt
# x = [1,2,3,4,5]
# y = [10,20,25,30,40]
# sizes = [100,200,300,400,1500]
# colors = [10,20,30,40,50]
# plt.scatter(x,y,s=sizes,c=colors,cmap="viridis")
# plt.colorbar(label="Color Scale")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Advanced Scatter Plot")
# plt.show()


# import matplotlib.pyplot as plt
# tasks = ["Task1","Task2","Task3","Task4"]
# status = [90,75,60,70]
# plt.barh(tasks,status,color="coral")
# plt.xlabel("Completion")
# plt.title("Progress Tracker")
# plt.show()





# import matplotlib.pyplot as plt
# students = ["A","B","C","D"]
# marks = [70,85,60,90]
# bars = plt.bar(students,marks,color="orange")

# for bar in bars:
#     plt.text(
#         bar.get_x() + bar.get_width()/2,
#         bar.get_height(),
#         bar.get_height(),
#         ha="center",
#         va="bottom"
#     )

# plt.title("Student Marks")
# plt.ylabel("Marks")
# plt.show()


# import matplotlib.pyplot as plt
# days = [1, 2, 3, 4, 5]
# sales = [200, 400, 600, 800, 1000]
# profit = [50, 150, 300, 500, 700]
# plt.plot(days, sales, label="Sales", marker="o")
# plt.plot(days, profit, label="Profit", marker="s")
# plt.xlabel("Days")
# plt.ylabel("Amount")
# plt.title("Sales vs Profit")
# plt.legend()
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt
# x = [1,2,3,4,5]
# y = [10,20,30,40,50]
# plt.plot(x,y,color="red",linestyle="--",linewidth=3,marker="o",markersize=10)
# plt.grid(True)
# plt.title("Custom Line Graph")
# plt.xlabel("X-axis")
# plt.ylabel("Y-label")
# plt.show()

# import matplotlib.pyplot as plt
# languages = ["Python","Java","C","JavaScript"]
# usage = [40,25,20,15]
# plt.pie(
#         usage,
#         labels=languages,
#         colors=["red","green","blue","yellow"],
#         explode=[0.1,0,0,0],
#         autopct="%1.1f%%",
#         shadow=True,
#         startangle=100)
# plt.title("Sample")
# plt.show()



# import matplotlib.pyplot as plt
# x = [1,2,3,4,5]
# y = [5,10,15,20,25]
# plt.scatter(x,y)
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Scatter Plot")
# plt.show()


# import matplotlib.pyplot as plt
# languages = ["Python","Java","C","JavaScript"]
# usage = [40,25,20,15]
# custom_colors = ["gold","green","pink","blue"];
# plt.pie(usage,labels=languages,autopct="%1.1f%%",colors=custom_colors,explode=[0.1,0,0,0.1])
# plt.title("Languages Market Share")
# plt.show()


# import matplotlib.pyplot as plt
# students = ["Std1","Std2","Std3","Std4","Std5"]
# marks = [70,80,90,60,70]
# plt.bar(students,marks)
# plt.xlabel("Students")
# plt.ylabel("Marks")
# plt.title("Students Marks")
# plt.show()



# import matplotlib.pyplot as plt
# x = [1,2,3,4,5]
# y = [10,20,30,40,50]
# plt.plot(x,y)
# plt.xlabel("x-values")
# plt.ylabel("y-values")
# plt.title("Sample Line Graph")
# plt.show()


# import matplotlib
# print(matplotlib.__version__)
