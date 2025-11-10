# importing modules and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.backends.backend_svg import svgProlog
#from mpl_toolkits.mplot3d.proj3d import transform
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import SVG, display

from svgpath2mpl import parse_path
from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.widgets import RadioButtons
from matplotlib.patches import Patch
vis_modus = 'Måned'


display(SVG(filename='cloud.svg'))


def legg_til_fargeforklaring(ax):
    terskler = [1300, 1700, 2500, 3200]
    labels = [
        f"<{terskler[0]} mm",
        f"{terskler[0]}-{terskler[1]-1} mm",
        f"{terskler[1]}-{terskler[2]-1} mm",
        f"{terskler[2]}-{terskler[3]-1}mm",
        f"≥ {terskler[3]}mm",
    ]

    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5) for c in colors]

    leg = ax.legend(
        handles,
        labels,
        title="Årsnedbør (fargekode)",
        loc="lower left",
        frameon=True,
        borderpad=0.6,
        labelspacing=0.5,
        handlelength=1.6,
        handletextpad=0.6,
    )

    try:
        leg._legend_box.align ="left"
    except Exception:
        pass
    return leg

def draw_the_map():
    # Accumulate all months to year
    axMap.cla()
    axMap.imshow(img, extent=(0, 13, 0, 10))

    doc = minidom.parse("cloud.svg")
    path_strings = [p.getAttribute('d') for p in doc.getElementsByTagName('path')]
    doc.unlink()

    svg_path = parse_path(path_strings[0])
    svg_path = svg_path.transformed(mtransforms.Affine2D().scale(1, -1))
    svg_path.vertices -= svg_path.vertices.mean(axis=0)
    svg_path.vertices += [-20, 14]

    df_year = df.groupby(['X', 'Y']).agg({'Nedbor': 'sum'}).reset_index()
    xr = df_year['X'].tolist()
    yr = df_year['Y'].tolist()
    nedborAar = df_year['Nedbor']
    ColorList = [color_from_nedbor(n) for n in nedborAar]
    axMap.scatter(xr, yr, c=ColorList, s=size_from_nedbor(nedborAar/12) * 2, marker=svg_path)
    labels = [label_from_nedbor(n) for n in nedborAar]
    for i, y in enumerate(xr):
        axMap.text(xr[i], yr[i], s=labels[i], color='white', fontsize=8, ha='center', va='center')
    legg_til_fargeforklaring(axMap)

def index_from_nedbor(x):
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4

def color_from_nedbor(nedbor):
    return colors[index_from_nedbor(nedbor)]
def size_from_nedbor(nedbor):
    return 350
def label_from_nedbor(nedbor):
    return str(int(nedbor / 100))
def month_to_quarter_data(y_pred):
    q_values = [
        np.sum(y_pred[0:3]),
        np.sum(y_pred[3:6]),
        np.sum(y_pred[6:9]),
        np.sum(y_pred[9:12])
    ]
    labels = ['K1', 'K2', 'K3', 'K4']
    return labels, q_values

def on_click(event) :
    global marked_point
    if event.inaxes != axMap:
        return

    marked_point = (event.xdata, event.ydata)
    x,y = marked_point

    vectors = []
    months = np.linspace(1,12,12)
    for mnd in months:
        vectors.append([x,y,mnd])
    AtPoint = np.vstack(vectors)
    # fitting the model, and predict for each month
    AtPointM = poly.fit_transform(AtPoint)
    y_pred = model.predict(AtPointM)
    aarsnedbor = sum(y_pred)
    axGraph.cla()
    draw_the_map()
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) - click rød er estimert")


    axMap.text(x, y, s=label_from_nedbor(aarsnedbor), color='white', fontsize=8, ha='center', va='center')
    axGraph.set_title(f"Nedbør per måned, Årsnedbør {int(aarsnedbor)} mm")

    doc = minidom.parse("cloud.svg")
    path_strings = [p.getAttribute('d') for p in doc.getElementsByTagName('path')]
    doc.unlink()

#svg pointer
    svg_path = parse_path(path_strings[0])
    svg_path = svg_path.transformed(mtransforms.Affine2D().scale(1, -1))
    svg_path.vertices -= svg_path.vertices.mean(axis=0)
    svg_path.vertices += [-20,14]

  #  colorsPred = [color_from_nedbor(nedbor * 12) for nedbor in y_pred]
    axMap.scatter(x, y, c=color_from_nedbor(aarsnedbor), s=size_from_nedbor(aarsnedbor) * 2, marker=svg_path)
  #  axGraph.bar(months, y_pred, color=colorsPred)

    if vis_modus == 'Måned':
        colorsPred = [color_from_nedbor(n * 12) for n in y_pred]
        axGraph.bar(months, y_pred, color=colorsPred)
        draw_label_and_ticks()
        axGraph.set_title(f"Nedbør per måned, Årsnedbør {int(aarsnedbor)} mm")

    else:
        q_labels, q_values = month_to_quarter_data(y_pred)
        colorsQ = [color_from_nedbor(n * 4 * 12) for n in q_values]
        axGraph.bar(np.arange(1, 5), q_values, tick_label=q_labels, color=colorsQ)
        axGraph.set_title(f"Nedbør per kvartal, Årsnedbør {int(aarsnedbor)} mm")

    plt.show()


def draw_label_and_ticks():
    xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axGraph.set_xticks(np.linspace(1, 12, 12))
    axGraph.set_xticklabels(xlabels)

# Create the figures
fig = plt.figure(figsize=(10, 4))
axGraph = fig.add_axes((0.05, 0.07, 0.35, 0.85))
axMap = fig.add_axes((0.41, 0.07, 0.59, 0.85))
draw_label_and_ticks()
img = mpimg.imread('StorBergen2.png')
axMap.set_title("Årsnedbør Stor Bergen")
axGraph.set_title("Per måned")
axMap.axis('off')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # Adjust the figure to fit the image
axMap.margins(x=0.01, y=0.01)  # Adjust x and y margins

axRadio = fig.add_axes((0.01, 0.94, 0.08, 0.06))
radio = RadioButtons(axRadio, ('Måned', 'Kvartal'))

def on_radio_change(label):
    global vis_modus
    vis_modus = label
    if marked_point != (0,0):
        class DummyEvent:
            def __init__(self, x, y):
                self.xdata = x
                self.ydata = y
                self.inaxes = axMap
        on_click(DummyEvent(*marked_point))
    else:
        draw_the_map()
        plt.draw()
radio.on_clicked(on_radio_change)

# Read rain data, and split in train and test.py data
df = pd.read_csv('NedborX.csv')
marked_point = (0,0)
ns = df['Nedbor']
X = df.drop('Nedbor',  axis=1)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_poly, ns, test_size=0.25)

# creating a regression model
model = LinearRegression()
model.fit(X_train, Y_train) # fitting the model
Y_pred = model.predict(X_test)

# Check model quality
r_squared = r2_score(Y_test, Y_pred)
print(f"R-squared: {r_squared:.2f}")
print('mean_absolute_error (mnd) : ', mean_absolute_error(Y_test, Y_pred))

colors = [ '#5dbcc6', '#458e96', '#356c72', '#23484c', '#172d30']
draw_the_map()

plt.connect('button_press_event', on_click)
plt.show()
legg_til_fargeforklaring(axMap)

