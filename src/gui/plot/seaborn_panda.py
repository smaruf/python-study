import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


df_obj1 = pd.DataFrame({"x": np.random.randn(500),
                   "y": np.random.randn(500)})
 
df_obj2 = pd.DataFrame({"x": np.random.randn(500),
                   "y": np.random.randint(0, 100, 500)})


sns.jointplot(x="x", y="y", data=df_obj2)
sns.jointplot(x="x", y="y", data=df_obj2, kind="hex");
sns.jointplot(x="x", y="y", data=df_obj1, kind="kde");
dataset = sns.load_dataset("tips")
sns.pairplot(dataset);

#titanic = sns.load_dataset('titanic')
#planets = sns.load_dataset('planets')
#flights = sns.load_dataset('flights')
#iris = sns.load_dataset('iris')
exercise = sns.load_dataset('exercise')
sns.stripplot(x="diet", y="pulse", data=exercise)
sns.swarmplot(x="diet", y="pulse", data=exercise, hue='kind')
sns.boxplot(x="diet", y="pulse", data=exercise)
sns.boxplot(x="diet", y="pulse", data=exercise, hue='kind')
sns.violinplot(x="diet", y="pulse", data=exercise, hue='kind')
sns.barplot(x="diet", y="pulse", data=exercise, hue='kind')
sns.pointplot(x="diet", y="pulse", data=exercise, hue='kind');

plt.show()
