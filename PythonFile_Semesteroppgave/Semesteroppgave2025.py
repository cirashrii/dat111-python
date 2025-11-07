# importing modules and packages
import pandas as pd
import numpy as np
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import SVG, display
from matplotlib.widgets import Button
from svgpath2mpl import parse_path
from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms



display(SVG(filename='cloud.svg'))

nedbor_visning = True

def bytt_til_nedbor(event):
    nedbor_visning = True

def bytt_til_temperatur(event):
    nedbor_visning = False


def draw_the_map():
    # Accumulate all months to year
    axMap.cla()
    plt.imshow(img, extent=(0, 13, 0, 10))

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

    colorsPred = [color_from_nedbor(nedbor * 12) for nedbor in y_pred]
    axMap.scatter(x, y, c=color_from_nedbor(aarsnedbor), s=size_from_nedbor(aarsnedbor) * 2, marker=svg_path)
    axGraph.bar(months, y_pred, color=colorsPred)
    gjennomsnitt = (aarsnedbor / 12)
    txt = "Gjennomsnitt:{:.2f}mm"
    axGraph.axhline(y=gjennomsnitt, xmin=0, xmax=1, color='#ea9d02', linestyle='-', linewidth=2, alpha=0.8)
    axGraph.text(x=0.2, y=(aarsnedbor / 12) + 3, s=txt.format(gjennomsnitt), fontsize=10, color='#ea9d02', alpha=1,
                 weight='bold')
    axGraph.grid(visible=True, which='major', axis='y', linestyle='-', linewidth=0.5, color='black')
    draw_label_and_ticks()
    plt.show()


def draw_label_and_ticks():
    xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axGraph.set_xticks(np.linspace(1, 12, 12))
    axGraph.set_xticklabels(xlabels)

# Create the figures
fig = plt.figure(figsize=(10, 6))
axGraph = fig.add_axes((0.05, 0.3, 0.35, 0.62))
axMap = fig.add_axes((0.41, 0.3, 0.59, 0.62))
draw_label_and_ticks()
img = mpimg.imread('StorBergen2.png')
axMap.set_title("Årsnedbør Stor Bergen")
axGraph.set_title("Per måned")
axMap.axis('off')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # Adjust the figure to fit the image
axMap.margins(x=0.01, y=0.01)  # Adjust x and y margins

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

plt.legend(
    loc='upper left',
    title='Test',
    fontsize='8',
    frameon=True,
    shadow=True
)


plt.connect('button_press_event', on_click)
plt.show()


#Button
#axButn1 = plt.axes((0.1, 0.1, 0.3, 0.1))
#btn1 = Button(    axButn1, label="nedbør", color='lightblue', hovercolor='tomato')
#btn1.on_clicked(bytt_til_nedbor)

#axButn2 = plt.axes((0.55, 0.1, 0.3, 0.1))
#btn2 = Button( axButn2, label="temperatur", color='orange', hovercolor='tomato')
#btn2.on_clicked(bytt_til_temperatur)







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
    axMap.scatter(xr, yr, c='#ffee56', s=size_from_temperatur(TemperaturAar / 12) * 1, marker='o')
    labels = [label_from_temperatur(t) for t in TemperaturAar/12]
    for i, y in enumerate(xr):
        axMap.text(xr[i], yr[i], s=labels[i], color='black', fontsize=8, ha='center', va='center')
    axMap.set_title(f"Årstemperatur Stor Bergen")

def index_from_temperaturr(x):
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4


def color_from_teperatur(temperatur):
    return colors[index_from_nedbor(temperatur)]


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
    axGraph.axhline(y=gjennomsnitt, xmin=0, xmax=1, color='#ea9d02', linestyle='-', linewidth=2, alpha=0.8)
    axGraph.text(x=0.2, y=(aarstemperatur / 12) + 3, s=txt.format(gjennomsnitt), fontsize=10, color='#ea9d02', alpha=1,
                 weight='bold')




    axMap.scatter(x, y, c='#efd051', s=size_from_nedbor(aarstemperatur) * 1.25, marker='o')
    axGraph.bar(months, y_pred, color='#ffee56')
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

draw_the_map()

plt.connect('button_press_event', on_click)
plt.show()


