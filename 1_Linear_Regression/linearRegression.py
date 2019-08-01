import pandas
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# wczytanie danych
df = pandas.read_csv('diamonds_single.csv', sep=';', header=0)
print(df.head())
print(df.shape)

# wizualizacja danych

df.plot.scatter(x='carat', y='price')
# plt.pyplot.show()

# podział danych
X_diam = df[['carat']]
Y_diam = df[['price']]
X_train, X_test, y_train, y_test = train_test_split(X_diam, Y_diam, random_state=1)

# przygotowanie modelu
regr = linear_model.LinearRegression()

# trenowanie modelu
regr.fit(X_train, y_train)
print(regr)

# predykcja na danych testowych
y_pred = regr.predict(X_test)

# wyniki

# jakie sa wspolczynniki i wyraz wolny a i b ze wzoru y = ax + b
print('wspolczynnik a=', regr.coef_)
print('wspolczynnik a=', regr.intercept_)

# jaki bład sredniokwadratowy
print('blad sredniokwadratoey', mean_squared_error(y_test, y_pred))

# wspólczynnik determinacji czyli R^2
print('R^2', regr.score(X_train, y_train))

# wzor z modelu to: y = 7752 * x - 2245

# wizualizacja na wspolnym wykresie
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.show()



