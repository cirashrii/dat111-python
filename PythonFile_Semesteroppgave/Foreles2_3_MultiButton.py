import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.widgets import Button

x = np.linspace(0,20, 100)
y = x * 0

fig = figure(figsize=(6,5))
ax1 = fig.add_axes((0.1,0.3,0.8,0.5))
ax1.title.set_text('Graphs')

line, = plt.plot(x, y, color='red')

def plot1(event):
    line.set_ydata(np.cos(x))
    ax1.set_ylim(-1, 1)
    plt.draw()

axButn1 = plt.axes((0.1, 0.1, 0.3, 0.1))
btn1 = Button(    axButn1, label="cos", color='lightblue', hovercolor='tomato')
btn1.on_clicked(plot1)

def plot2(event):
    line.set_ydata(np.sin(x)*np.exp(-x*.1))
    ax1.set_ylim(-1, 1)
    plt.draw()

axButn2 = plt.axes((0.6, 0.1, 0.3, 0.1))
btn2 = Button( axButn2, label="sin(x)*exp(.1x)", color='orange', hovercolor='tomato')
btn2.on_clicked(plot2)

plt.show()