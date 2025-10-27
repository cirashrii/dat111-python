from matplotlib import pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np


def to_rgb(value):
    if max(0, min(100, value)) < 50:
        return value / 50, 1, 0
    return 1.0, 1 - (value - 50) / 50, 0


df: DataFrame = pd.read_csv('house_data.csv')
# Prepare plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes((.1, .2, .85, .75), projection="3d")
ax.set_title("Leiligheter i Bergen (Est/Real)")

# Split the dataset
X = df[['m2', 'Standard']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Standard avvik:{mae:.0f}")

x3 = np.linspace(25, 250, 10).tolist()
y3 = [1, 1, 2, 2, 3, 3, 4, 4, 1, 2]
z3 = np.zeros(10)

# størrelse på bar'ene
dx = np.ones(10) * 10
dy = np.ones(10) * 0.3
dz = [1, 2, 3, 4, 5, 6, 7, 8, 9.5, 10.2]

colors = [to_rgb(z / max(dz) * 100) for z in dz]
ax.bar3d(x3, y3, z3, dx, dy, dz, color=colors)

for i in range(len(x3)):
    ax.text(x=x3[i], y=y3[i], z=dz[i], s=f'{dz[i]:.1f}', fontsize=14, color='blue')

ax.set_ylim(1, 4)
ax.set_xlabel('m2')
ax.set_ylabel('standard')
ax.set_zlabel('pris')

plt.show()
