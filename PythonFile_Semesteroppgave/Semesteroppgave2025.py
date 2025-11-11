# importing modules and packages

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.backends.backend_svg import svgProlog
#from mpl_toolkits.mplot3d.proj3d import transform
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import SVG, display
from svgpath2mpl import parse_path
from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.widgets import Button
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

    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5) for c in nedborColors]

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




def legg_til_fargeforklaringTemp(ax):
    terskler = [10, 15, 20, 40]
    labels = [
        f"<{terskler[0]} C",
        f"{terskler[0]}-{terskler[1]-1} C",
        f"{terskler[1]}-{terskler[2]-1} C",
        f"{terskler[2]}-{terskler[3]-1}C",
        f"≥ {terskler[3]}C",
    ]

    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5) for c in tempColors]

    leg = ax.legend(
        handles,
        labels,
        title="Temperatur (fargekode)",
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



def draw_the_map_nedbor():
    # Accumulate all months to year
    axMapNedbor.cla()
    axMapNedbor.imshow(img, extent=(0, 13, 0, 10))

    doc = minidom.parse("cloud.svg")
    path_strings = [p.getAttribute('d') for p in doc.getElementsByTagName('path')]
    doc.unlink()

    svg_path = parse_path(path_strings[0])
    svg_path = svg_path.transformed(mtransforms.Affine2D().scale(1, -1))
    svg_path.vertices -= svg_path.vertices.mean(axis=0)
    svg_path.vertices += [-20, 14]

    df_year = df_n.groupby(['X', 'Y']).agg({'Nedbor': 'sum'}).reset_index()
    xr = df_year['X'].tolist()
    yr = df_year['Y'].tolist()
    nedborAar = df_year['Nedbor']
    ColorList = [color_from_nedbor(n) for n in nedborAar]
    axMapNedbor.scatter(xr, yr, c=ColorList, s=size_from_nedbor(nedborAar/12) * 2, marker=svg_path)
    labels = [label_from_nedbor(n) for n in nedborAar]
    for i, y in enumerate(xr):
        axMapNedbor.text(xr[i], yr[i], s=labels[i], color='white', fontsize=8, ha='center', va='center')
    axMapNedbor.set_title(f"Årsnedbør Stor Bergen")
    legg_til_fargeforklaring(axMapNedbor)

def draw_the_map_temperatur():
    # Accumulate all months to year
    axMapTemperatur.cla()
    axMapTemperatur.imshow(img, extent=(0, 13, 0, 10))


    df_year = df_t.groupby(['X', 'Y']).agg({'Temperatur': 'sum'}).reset_index()
    xr = df_year['X'].tolist()
    yr = df_year['Y'].tolist()
    TemperaturAar = df_year['Temperatur']
    ColorList = [color_from_temperatur(t) for t in TemperaturAar/12]



    axMapTemperatur.scatter(xr, yr, c=ColorList, s=size_from_temperatur(TemperaturAar / 12) * 1, marker='o')
    labels = [label_from_temperatur(t) for t in TemperaturAar/12]
    for i, y in enumerate(xr):
        axMapTemperatur.text(xr[i], yr[i], s=labels[i], color='black', fontsize=8, ha='center', va='center')
    axMapTemperatur.set_title(f"Gjennomsnittstemperatur Stor Bergen")

    legg_til_fargeforklaringTemp(axMapTemperatur)


def index_from_nedbor(x):
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4

def color_from_nedbor(nedbor):
    return nedborColors[index_from_nedbor(nedbor)]
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
    labels = ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Okt-Des']
    return labels, q_values

def on_click_nedbor(event) :
    global marked_point
    if event.inaxes != axMapNedbor:
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
    y_pred = n_model.predict(AtPointM)
    aarsnedbor = sum(y_pred)
    axGraphManed.cla()
    axGraphKvartal.cla()
    draw_the_map_nedbor()


    axMapNedbor.text(x, y, s=label_from_nedbor(aarsnedbor), color='white', fontsize=8, ha='center', va='center')
    axGraphManed.set_title(f"Nedbør per måned, Årsnedbør {int(aarsnedbor)} mm")

    doc = minidom.parse("cloud.svg")
    path_strings = [p.getAttribute('d') for p in doc.getElementsByTagName('path')]
    doc.unlink()

    #svg pointer
    svg_path = parse_path(path_strings[0])
    svg_path = svg_path.transformed(mtransforms.Affine2D().scale(1, -1))
    svg_path.vertices -= svg_path.vertices.mean(axis=0)
    svg_path.vertices += [-20,14]

    colorsPred = [color_from_nedbor(nedbor * 12) for nedbor in y_pred]
    axMapNedbor.scatter(x, y, c=color_from_nedbor(aarsnedbor), s=size_from_nedbor(aarsnedbor) * 2, marker=svg_path)
    axGraphManed.bar(months, y_pred, color=colorsPred)
    gjennomsnittManed = (aarsnedbor / 12)
    txt = "Gjennomsnitt:{:.2f}mm"
    axGraphManed.axhline(y=gjennomsnittManed, xmin=0, xmax=1, color='#ea9d02', linestyle='-', linewidth=2, alpha=0.8)
    axGraphManed.text(x=0.2, y=(aarsnedbor / 12) + 3, s=txt.format(gjennomsnittManed), fontsize=10, color='#ea9d02',
                      alpha=1,
                      weight='bold')
    draw_label_and_ticks_maned()

    q_labels, q_values = month_to_quarter_data(y_pred)
    colorsQ = [color_from_nedbor(n * 4) for n in q_values]
    axGraphKvartal.bar(np.arange(1, 5), q_values, tick_label=q_labels, color=colorsQ)
    axGraphKvartal.set_title(f"Nedbør per kvartal, Årsnedbør {int(aarsnedbor)} mm")
    gjennomsnittKvartal = (aarsnedbor / 4)
    txt = "Gjennomsnitt:{:.2f}mm"
    axGraphKvartal.axhline(y=gjennomsnittKvartal, xmin=0, xmax=1, color='#ea9d02', linestyle='-', linewidth=2, alpha=0.8)
    axGraphKvartal.text(x=0.45, y=(aarsnedbor / 4) + 5, s=txt.format(gjennomsnittKvartal), fontsize=10, color='#ea9d02',
                 alpha=1,
                 weight='bold')

    plt.draw()
    legg_til_fargeforklaring(axMapNedbor)

def on_click_temp(event):
    global marked_point
    if event.inaxes != axMapTemperatur:
        return

    marked_point = (event.xdata, event.ydata)
    x, y = marked_point

    vectors = []
    months = np.linspace(1, 12, 12)
    for mnd in months:
        vectors.append([x, y, mnd])
    AtPoint = np.vstack(vectors)
    # fitting the model, and predict for each month
    AtPointM = poly.fit_transform(AtPoint)
    y_pred = t_model.predict(AtPointM)
    aarstemperatur = sum(y_pred)/12
    axGraphManedTemp.cla()
    axGraphKvartalTemp.cla()
    draw_the_map_temperatur()


    axMapTemperatur.text(x, y, s=label_from_temperatur(aarstemperatur), color='black', fontsize=8, ha='center', va='center')
    axGraphManedTemp.set_title(f"Temperatur per måned")

    gjennomsnitt = (aarstemperatur)
    txt = "Gjennomsnitt:{:.2f}C"
    axGraphManedTemp.axhline(y=gjennomsnitt, xmin=0, xmax=1, color='#b70707', linestyle='-', linewidth=2, alpha=0.8)
    axGraphManedTemp.text(x=0.2, y=(aarstemperatur)+0.5, s=txt.format(gjennomsnitt), fontsize=10, color='#b70707', alpha=1,
                 weight='bold')



    colorsPred = [color_from_temperatur(temperatur) for temperatur in y_pred]

    axMapTemperatur.scatter(x, y, c=color_from_temperatur(aarstemperatur), s=size_from_temperatur(aarstemperatur) * 1.25, marker='o')
    axGraphManedTemp.bar(months, y_pred, color=colorsPred)
    draw_label_and_ticks_maned()

    q_labels, q_values = month_to_quarter_data(y_pred/3)
    colorsQ = [color_from_temperatur(t) for t in q_values]
    axGraphKvartalTemp.bar(np.arange(1, 5), q_values, tick_label=q_labels, color=colorsQ)
    axGraphKvartalTemp.set_title(f"Temperatur per kvartal")
    gjennomsnittKvartal = (aarstemperatur)
    txt = "Gjennomsnitt:{:.2f}C"
    axGraphKvartalTemp.axhline(y=gjennomsnittKvartal, xmin=0, xmax=1, color='#b70707', linestyle='-', linewidth=2,
                           alpha=0.8)
    axGraphKvartalTemp.text(x=0.45, y=(aarstemperatur) + 0.25, s=txt.format(gjennomsnittKvartal), fontsize=10, color='#b70707',
                        alpha=1,
                        weight='bold')
    legg_til_fargeforklaringTemp(axMapTemperatur)
    plt.draw()

def index_from_temperatur(x):
    if x < 10: return 0
    if x < 15: return 1
    if x < 20: return 2
    if x < 40: return 3
    return 4


def color_from_temperatur(temperatur):
    return tempColors[index_from_temperatur(temperatur)]


def size_from_temperatur(temperatur):
    return 350


def label_from_temperatur(temperatur):
    return str(int(temperatur))


def draw_label_and_ticks_maned():
    xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axGraphManed.set_xticks(np.linspace(1, 12, 12))
    axGraphManed.set_xticklabels(xlabels)
    axGraphManedTemp.set_xticks(np.linspace(1, 12, 12))
    axGraphManedTemp.set_xticklabels(xlabels)

# Create the figures
fig = plt.figure(figsize=(10, 6))
axGraphManed = fig.add_axes((0.05, 0.17, 0.35, 0.67))
axGraphKvartal = fig.add_axes((0.05, 0.17, 0.35, 0.67))
axGraphKvartal.set_visible(False)
axMapNedbor = fig.add_axes((0.41, 0.17, 0.59, 0.67))

axGraphManedTemp = fig.add_axes((0.05, 0.17, 0.35, 0.67))
axGraphKvartalTemp = fig.add_axes((0.05, 0.17, 0.35, 0.67))
axGraphKvartalTemp.set_visible(False)
axGraphManedTemp.set_visible(False)
axMapTemperatur = fig.add_axes((0.41, 0.17, 0.59, 0.67))
axMapTemperatur.set_visible(False)

draw_label_and_ticks_maned()
img = mpimg.imread('StorBergen2.png')
axMapNedbor.set_title("Årsnedbør Stor Bergen")
axGraphManed.set_title("Per måned")
axGraphKvartal.set_title("Per kvartal")
axMapNedbor.axis('off')

axMapTemperatur.set_title("Årstemperatur Stor Bergen")
axGraphManedTemp.set_title("Per måned")
axGraphKvartalTemp.set_title("Per kvartal")
axMapTemperatur.axis('off')


fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # Adjust the figure to fit the image
axMapNedbor.margins(x=0.01, y=0.01)  # Adjust x and y margins


def manadsvisning(event):
    axGraphManed.set_visible(True)
    axGraphKvartal.set_visible(False)
    fig.canvas.draw_idle()

def kvartalvisning(event):
    axGraphManed.set_visible(False)
    axGraphKvartal.set_visible(True)
    fig.canvas.draw_idle()

def manadsvisningTemp(event):
    axGraphManedTemp.set_visible(True)
    axGraphKvartalTemp.set_visible(False)
    fig.canvas.draw_idle()

def kvartalvisningTemp(event):
    axGraphManedTemp.set_visible(False)
    axGraphKvartalTemp.set_visible(True)
    fig.canvas.draw_idle()

# Read rain data, and split in train and test.py data
marked_point = (0,0)

df_n = pd.read_csv('NedborX.csv')
df_t = pd.read_csv('TemperaturX.csv')

ns_n = df_n['Nedbor']
X_n = df_n.drop('Nedbor',  axis=1)

poly = PolynomialFeatures(degree=3)
X_n_poly = poly.fit_transform(X_n)
X_n_train, X_n_test, Y_n_train, Y_n_test = train_test_split(
    X_n_poly, ns_n, test_size=0.25)

ns_t = df_t['Temperatur']
X_t = df_t.drop('Temperatur', axis=1)

X_t_poly = poly.fit_transform(X_t)
X_t_train, X_t_test, Y_t_train, Y_t_test = train_test_split(
    X_t_poly, ns_t, test_size=0.25)

# creating a regression model
n_model = LinearRegression()
n_model.fit(X_n_train, Y_n_train) # fitting the model
Y_n_pred = n_model.predict(X_n_test)

# Check model quality
r_squared_n = r2_score(Y_n_test, Y_n_pred)
print(f"R-squared: {r_squared_n:.2f}")
print('mean_absolute_error (mnd) : ', mean_absolute_error(Y_n_test, Y_n_pred))

t_model = LinearRegression()
t_model.fit(X_t_train, Y_t_train) # fitting the model
Y_t_pred = t_model.predict(X_t_test)

r_squared_t = r2_score(Y_t_test, Y_t_pred)
print(f"R-squared: {r_squared_t:.2f}")
print('mean_absolute_error (mnd) : ', mean_absolute_error(Y_t_test, Y_t_pred))

nedborColors = [ '#5dbcc6', '#458e96', '#356c72', '#23484c', '#172d30']
tempColors = [ '#fcf341', '#f9c32c', '#ef7e04', '#d14545', '#560505']
draw_the_map_nedbor()
draw_the_map_temperatur()


plt.connect('button_press_event', on_click_nedbor)
plt.connect('button_press_event', on_click_temp)

def nedbor_show(event):
    axGraphKvartalTemp.set_visible(False)
    axMapNedbor.set_visible(True)
    axMapTemperatur.set_visible(False)
    axGraphManed.set_visible(True)
    axGraphManedTemp.set_visible(False)
    print("Nedbør skal vises!")
    axButnKvartalTemp.set_visible(False)
    axButnTemperaturManad.set_visible(False)
    axButnNedborKvartal.set_visible(True)
    axButnNedborManad.set_visible(True)

    btn_temperatur_manad.set_active(False)
    btn_temperatur_kvartal.set_active(False)
    btn_nedbor_manad.set_active(True)
    btn_nedbor_kvartal.set_active(True)

    fig.canvas.draw_idle()


def temperatur_show(event):
    axGraphKvartal.set_visible(False)
    axMapNedbor.set_visible(False)
    axMapTemperatur.set_visible(True)
    axGraphManed.set_visible(False)
    axGraphManedTemp.set_visible(True)
    axButnNedborKvartal.set_visible(False)
    axButnNedborManad.set_visible(False)
    axButnKvartalTemp.set_visible(True)
    axButnTemperaturManad.set_visible(True)

    btn_temperatur_manad.set_active(True)
    btn_temperatur_kvartal.set_active(True)
    btn_nedbor_manad.set_active(False)
    btn_nedbor_kvartal.set_active(False)

    fig.canvas.draw_idle()


axButnNedborManad = plt.axes((0.05, 0.9, 0.167, 0.05))
btn_nedbor_manad = Button(axButnNedborManad, label="Månedsvisning", color='lightblue', hovercolor='tomato')
btn_nedbor_manad.on_clicked(manadsvisning)

axButnNedborKvartal = plt.axes((0.233, 0.9, 0.167, 0.05))
btn_nedbor_kvartal = Button(axButnNedborKvartal, label="Kvartalvisning", color='lightblue', hovercolor='tomato')
btn_nedbor_kvartal.on_clicked(kvartalvisning)

axButn1 = plt.axes((0.515, 0.9, 0.167, 0.05))
btn1 = Button(axButn1, label="Nedbør", color='lightblue', hovercolor='tomato')
btn1.on_clicked(nedbor_show)

axButn2 = plt.axes((0.735, 0.9, 0.167, 0.05))
btn2 = Button( axButn2, label="Temperatur", color='orange', hovercolor='tomato')
btn2.on_clicked(temperatur_show)

axButnTemperaturManad = plt.axes((0.05, 0.9, 0.167, 0.05))
btn_temperatur_manad = Button(axButnTemperaturManad, label="Månedsvisning", color='orange', hovercolor='tomato')
btn_temperatur_manad.on_clicked(manadsvisningTemp)
axButnTemperaturManad.set_visible(False)

axButnKvartalTemp = plt.axes((0.233, 0.9, 0.167, 0.05))
btn_temperatur_kvartal = Button(axButnKvartalTemp, label="Kvartalvisning", color='orange', hovercolor='tomato')
btn_temperatur_kvartal.on_clicked(kvartalvisningTemp)
axButnKvartalTemp.set_visible(False)




plt.show()


