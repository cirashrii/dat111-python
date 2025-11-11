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

fig = plt.figure(figsize=(6, 3.5))

axTextbox = plt.axes((0.5, 0.3, 0.3, 0.4))
axTextbox.axis('off')
plt.text(0,1,"Her kan du finne oversikten over \nnedbøren og temperaturen i Bergen.\nTrykk på relevant knapp for å se grafen.\nKos deg :)  -Gruppe 1",
         fontsize="16",
         ha="center",
         va="center")

def plot1(event):
    print("Plot 1 skal vises!")
    plt.close()
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
        axMap.set_title(f"Årsnedbør Stor Bergen")
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
        labels = ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Okt-Des']
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
        axGraphManed.cla()
        axGraphKvartal.cla()
        draw_the_map()


        axMap.text(x, y, s=label_from_nedbor(aarsnedbor), color='white', fontsize=8, ha='center', va='center')
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
        axMap.scatter(x, y, c=color_from_nedbor(aarsnedbor), s=size_from_nedbor(aarsnedbor) * 2, marker=svg_path)
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
        legg_til_fargeforklaring(axMap)

    def draw_label_and_ticks_maned():
        xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        axGraphManed.set_xticks(np.linspace(1, 12, 12))
        axGraphManed.set_xticklabels(xlabels)

    # Create the figures
    fig = plt.figure(figsize=(10, 6))
    axGraphManed = fig.add_axes((0.05, 0.17, 0.35, 0.67))
    axGraphKvartal = fig.add_axes((0.05, 0.17, 0.35, 0.67))
    axGraphKvartal.set_visible(False)
    axMap = fig.add_axes((0.41, 0.17, 0.59, 0.67))
    draw_label_and_ticks_maned()
    img = mpimg.imread('StorBergen2.png')
    axMap.set_title("Årsnedbør Stor Bergen")
    axGraphManed.set_title("Per måned")
    axGraphKvartal.set_title("Per kvartal")
    axMap.axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # Adjust the figure to fit the image
    axMap.margins(x=0.01, y=0.01)  # Adjust x and y margins

    def manadsvisning(event):
        axGraphManed.set_visible(True)
        axGraphKvartal.set_visible(False)
        print("Månedsvisning skal vises!")
        fig.canvas.draw_idle()

    def kvartalvisning(event):
        axGraphManed.set_visible(False)
        axGraphKvartal.set_visible(True)
        print("Kvartalsvisning skal vises!")
        fig.canvas.draw_idle()

    axButnNedborManad = plt.axes((0.05, 0.9, 0.167, 0.05))
    btn_nedbor_manad = Button(axButnNedborManad, label="Månedsvisning", color='lightblue', hovercolor='tomato')
    btn_nedbor_manad.on_clicked(manadsvisning)

    axButnNedborKvartal = plt.axes((0.233, 0.9, 0.167, 0.05))
    btn_nedbor_kvartal = Button(axButnNedborKvartal, label="Kvartalvisning", color='lightblue', hovercolor='tomato')
    btn_nedbor_kvartal.on_clicked(kvartalvisning)

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


#Button
axButn1 = plt.axes((0.1, 0.2, 0.3, 0.15))
btn1 = Button(    axButn1, label="Nedbør", color='lightblue', hovercolor='tomato')
btn1.label.set_fontsize(14)
btn1.on_clicked(plot1)

def plot2(event):
    # Read temp data, and split in train and test.py data
    df = pd.read_csv('TemperaturX.csv')
    marked_point = (0,0)
    ns = df['Temperatur']
    X = df.drop('Temperatur',  axis=1)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_poly, ns, test_size=0.25)

    # creating a regression model
    model = LinearRegression()
    model.fit(X_train, Y_train) # fitting the model
    Y_pred = model.predict(X_test)

    def draw_the_map():
        # Accumulate all months to year
        axMap.cla()
        plt.imshow(img, extent=(0, 13, 0, 10))

        df_year = df.groupby(['X', 'Y']).agg({'Temperatur': 'sum'}).reset_index()
        xr = df_year['X'].tolist()
        yr = df_year['Y'].tolist()
        TemperaturAar = df_year['Temperatur']
        ColorList = [color_from_temperatur(t) for t in TemperaturAar/12]



        axMap.scatter(xr, yr, c=ColorList, s=size_from_temperatur(TemperaturAar / 12) * 1, marker='o')
        labels = [label_from_temperatur(t) for t in TemperaturAar/12]
        for i, y in enumerate(xr):
            axMap.text(xr[i], yr[i], s=labels[i], color='black', fontsize=8, ha='center', va='center')
        axMap.set_title(f"Gjennomsnittstemperatur Stor Bergen")

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


    def on_click(event):
        global marked_point
        if event.inaxes != axMap:
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
        y_pred = model.predict(AtPointM)
        aarstemperatur = sum(y_pred)/12
        axGraph.cla()
        draw_the_map()


        axMap.text(x, y, s=label_from_temperatur(aarstemperatur), color='black', fontsize=8, ha='center', va='center')
        axGraph.set_title(f"Temperatur per måned, Årstemperatur {int(aarstemperatur)}")

        gjennomsnitt = (aarstemperatur)
        txt = "Gjennomsnitt:{:.2f}C"
        axGraph.axhline(y=gjennomsnitt, xmin=0, xmax=1, color='#b70707', linestyle='-', linewidth=2, alpha=0.8)
        axGraph.text(x=0.2, y=(aarstemperatur)+0.5, s=txt.format(gjennomsnitt), fontsize=10, color='#b70707', alpha=1,
                     weight='bold')



        colorsPred = [color_from_temperatur(temperatur) for temperatur in y_pred]

        axMap.scatter(x, y, c=color_from_temperatur(aarstemperatur), s=size_from_temperatur(aarstemperatur) * 1.25, marker='o')
        axGraph.bar(months, y_pred, color=colorsPred)
        draw_label_and_ticks()
        plt.draw()



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
    axMap.set_title("Årstemperatur Stor Bergen")
    axGraph.set_title("Per måned")
    axMap.axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the figure to fit the image
    axMap.margins(x=0.01, y=0.01)  # Adjust x and y margins

    tempColors = [ '#fcf341', '#f9c32c', '#ef7e04', '#d14545', '#560505']

    draw_the_map()

    plt.connect('button_press_event', on_click)
    plt.show()

axButn2 = plt.axes((0.55, 0.2, 0.3, 0.15))
btn2 = Button( axButn2, label="Temperatur", color='orange', hovercolor='tomato')
btn2.label.set_fontsize(14)
btn2.on_clicked(plot2)

plt.show()


