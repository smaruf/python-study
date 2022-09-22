import matplotlib.pyplot as plt
import seaborn as sns, numpy as np
from pylab import *

sns.set(rc={"figure.figsize": (8, 4)}); np.random.seed(0)
x = np.random.randn(100)

subplot(2,2,1)
ax = sns.distplot(x)

subplot(2,2,2)
ax = sns.distplot(x, rug=False, hist=False)

subplot(2,2,3)
ax = sns.distplot(x, vertical=True)

subplot(2,2,4)
ax = sns.kdeplot(x, shade=True, color="r")

plt.show()
