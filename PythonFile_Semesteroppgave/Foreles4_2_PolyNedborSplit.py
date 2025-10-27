
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data (rain!)
np.random.seed(1234)
X = np.linspace(0, 20, 41).reshape(-1, 1) # [0,0.5,1,1.5 ... 20]
y = 1500 + np.random.randint(-150, 450, size=X.shape) + pow(X-12, 2)*7

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Generate polynominal matrix/table
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
X_all_poly = poly.transform(X)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)  # y = Ax + Bx2..Cx3 -> calulate A, B, C...

# Predict, find the curve
y_pred = model.predict(X_test_poly)
# Evaluate model quality
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

# Plot
plt.scatter(X, y, color='blue', label='Train data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.scatter(X_test, y_pred, color='red', label='Pred data')

plt.plot(X, model.predict(X_all_poly), color='red', label='Regression line')
plt.xlabel('km from Bergen')
plt.ylabel('Rain (mm)')
plt.title(f'Linear Regression STD: {np.sqrt(mse_test):.2f}, R2: {r2_test:.2f}')
plt.legend()
plt.show()





