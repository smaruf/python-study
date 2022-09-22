import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)
y1 = 2*x + 1
y2 = 2**x + 1

plt.figure()
plt.plot(x, y1)  

plt.xlabel("I am x")
plt.ylabel("I am y")
plt.title("With Labels")

plt.show()
