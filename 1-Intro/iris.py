# Load the Iris data set into a pandas data frame.
# Drop all rows where the septal length is great than 5.  
# Scatter plot petal v. septal length with each dot color-coded by the species of iris.
# Add comments explaining each step in real terms: i.e., explain to a reader what and iris is, what a septal length is, what the units on the numbers are and what your filtering script is accomplishing.
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# load iris dataset
iris = datasets.load_iris()
print(iris.feature_names)
df = pd.DataFrame(iris.data, columns=iris.feature_names) # turn into a pandas dataframe
df["species"] = iris.target_names[iris.target] # add species column

# drop all the rows with sepal length over 5
df = df[(df["sepal length (cm)"] <= 5)]
df.head()

# scatter plot different species of iris
plt.figure(figsize = (7, 5))
ax = sns.scatterplot(data=df,x="petal length (cm)",y="sepal length (cm)", hue="species")
plt.show()

