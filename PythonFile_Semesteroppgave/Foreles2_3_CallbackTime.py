import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datetime import  datetime as dt

fig = plt.figure(figsize=(4,3))
def on_getTime(event):
    text = f"Time={dt.now().hour:0>2}:{dt.now().minute:0>2}:{dt.now().second:0>2}:{dt.now().microsecond:0>6}"
    axtime.set_title(text)
    plt.draw()
axtime = fig.add_axes((0.40, 0.05, 0.3, 0.2))
buttonTime = Button(axtime, label="Update time", hovercolor='yellow')
buttonTime.on_clicked(on_getTime)
plt.show()