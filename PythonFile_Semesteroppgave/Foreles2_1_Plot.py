
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,10) # array/tabell. Mye brukt
print(x)
y = np.sin(x)
fig = plt.figure(figsize=(10, 4))

axis = fig.add_axes(rect=(.1, .1, .8, .8))
axis.set_title("y = sin(x)")
axis.set_xlabel("x verdier")
axis.set_ylabel("y verdier")
axis.plot(x, y)
plt.show()

