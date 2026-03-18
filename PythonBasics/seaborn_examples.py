import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
data = {
    "Age" : [22,25,30,35,40,45],
    "Salary" : [25000,30000,50000,60000,75000,90000],
    "Department":["IT","HR","IT","Finance","HR","Finance"]
}
df = pd.DataFrame(data)
sns.set_style("whitegrid")
# sns.scatterplot(
#     x="Age",
#     y="Salary",
#     data=df
# )
# plt.title("Age Vs Salary")
# plt.show()

# sns.lineplot(x="Age",y="Salary",data=df)
# plt.title("Salary Vs Age")
# plt.show()

# sns.barplot(x="Department",y="Salary",data=df)
# plt.title("Average Salary")
# plt.show()

# sns.countplot(x="Department",data=df)
# plt.title("Employee Count per Department")
# plt.show()

# sns.histplot(df["Salary"],bins=5,kde=True)
# plt.title("Salary Distribution")
# plt.show()

# sns.boxplot(x="Department",y="Salary",data=df)
# plt.title("Salary By Department")
# plt.show()

# sns.violinplot(x="Department",y="Salary",data=df)
# plt.title("Salary Distribution Shape")
# plt.show()

corr = df[["Age","Salary"]].corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()








