# Script 2: Load, clean, and visualize the MNIST handwritten digits data set.
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
df = pd.DataFrame(digits.data) # Load the MNIST data set into a pandas data frame.
df["label"] = digits.target

df = df[df["label"].isin([3, 8])] # Drop all digits except 3 and 8.
# Visualize an array of MNIST digits so you can see some of the variation in the writing.
sample = df.sample(40)  # random sample
 
fig, axes = plt.subplots(5, 8, figsize=(10, 7))
for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
    image = row.drop("label").to_numpy().reshape(8, 8)  # 64 pixels = 8x8
    ax.imshow(image, cmap="gray_r")
    ax.set_title(int(row["label"]), fontsize=9)
    ax.axis("off") 
 
plt.show()
