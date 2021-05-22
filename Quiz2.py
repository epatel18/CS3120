import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

m = 500
np.random.seed(seed=5)
X = 6 * np.random.random(m).reshape(-1, 1) - 3 
y = 0.5 * X**5 - X**3 - X**2 + 2 + 5 * np.random.randn(m, 1)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.40, random_state=0)

reg1 = LinearRegression()
reg1.fit(X_train, y_train)
y_pred1 = reg1.predict(X_test) 

reg2 = LinearRegression()
poly2_features = PolynomialFeatures(degree=2)
X_poly2 = poly2_features.fit_transform(X_train)
X_poly2_test = poly2_features.fit_transform(X_test)
reg2.fit(X_poly2, y_train)
y_pred2 = reg2.predict(X_poly2_test)

score1 = mean_squared_error(y_test, y_pred1) 
score2 = mean_squared_error(y_test, y_pred2) 
print("Linear regression loss: ", score1) 
print("Polynomial loss: ", score2) 

plt.plot(X_train, y_pred1, color='blue')
plt.plot(X_train, y_pred2, color='green')
plt.title('green-poly, blue-linear, degree: 20')
plt.show()