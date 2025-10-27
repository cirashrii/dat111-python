from random import randint

import matplotlib.pyplot as py
import numpy as np

mnd = np.linspace(1,365,12)
nedbor = np.random.randint(150,450, size = mnd.shape)
fig = py.figure(figsize=(10,4))
ax = fig.add_axes((0.1,0.1, 0.4,0.8))
ax.bar(mnd, nedbor, width=20)
ax.scatter(mnd,nedbor,c="red")
#ax.plot(mnd,nedbor)
labels = ['J','F','M','A','M','J', 'J', 'A', 'S', 'O', 'N', 'D']
ax.set_ylim([100, 600])
ax.set_xticks(mnd)
ax.set_xticklabels(labels)
ax.set_title('Nedbør i Bergen (per mnd)')

py.show()