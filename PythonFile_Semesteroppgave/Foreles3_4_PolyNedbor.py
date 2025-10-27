
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(1234)
XKm = np.linspace(0, 20, 21).reshape(-1, 1) # [0,1,2..20]
y = 1500 + np.random.randint(-150, 450, size=XKm.shape) + pow(XKm-12, 2)*7
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(XKm) # make tables for all x,x2,x3....

# Train the model
model = LinearRegression()
model.fit(X_poly, y)  # y = Ax + Bx2..Cx3 -> calulate A, B, C...

# Predict, find the curve
y_pred = model.predict(X_poly) # calulate y = Ax + Bx2..
std = np.sqrt(mean_squared_error(y, y_pred)) # standard deviation/avvk
r_score = r2_score(y, y_pred)

# Plot
plt.scatter(XKm, y, color='blue', label='Train data')
plt.scatter(XKm, y_pred, color='red', label='Estimated')
plt.plot(XKm, y_pred, color='red', label='Regression line')
plt.xlabel('km from Bergen')
plt.ylabel('Rain (mm)')
plt.title(f'Linear Regression STD: {std:.2f}, R2: {r_score:.2f}')
plt.legend()
plt.show()





