import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# wczytanie danych
df = pd.read_csv('Position_Salaries.csv', sep=',', header=0)
print(df)
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
print('X=', X)
print('y=', y)

# podzial danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('aaa')
print(X_train)
print(X_test)


# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print('bbb')
print('Feature Scaling X_train', X_train)
print('Feature Scaling X_test', X_test)

# wybranie i uczenie modelu regresji liniowej
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(X, y)

# wybranie i uczenie modelu regresji welomianowej i liniowej
from sklearn.preprocessing import PolynomialFeatures

poly_regr = PolynomialFeatures(degree=4)
X_poly = poly_regr.fit_transform(X)
poly_regr.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# wizualizacja regresji liniowej
import matplotlib.pyplot as plt
plt.scatter(X, y, color='red')
plt.plot(X, lin_regr.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# wizualizacja regresji wielomianowej

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_regr.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# wizualizacja regresji wielomianowej  (lepsza rozdzielczosc)
import numpy as np
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_regr.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()