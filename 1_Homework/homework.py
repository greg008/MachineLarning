import pandas

# wczytanie danych
df = pandas.read_csv('Zadanie-domowe-50_Startups.csv', sep=',', header=0)
print(df.head())

# przygotowanie danych
# dodanie dummy variables zamiast state i usuniecie ze wzgledu na multicoliteraly trap

def addColumnNy(row):
    if row['State'] == 'New York':
        return 1
    else:
        return 0

def addColumnFl(row):
    if row['State'] == 'Florida':
        return 1
    else:
        return 0

# apply stasuje do kazdego wiersza a axis 1 oznacza ze do kolumn czyli osi pionowej

df['NY'] = df.apply(addColumnNy, axis=1)
df['FL'] = df.apply(addColumnFl, axis=1)
print(df)

print(df.head())
newdf = df.drop(['State'], axis=1)
print(newdf.head())

# podział na dane treningowe i testowe
from sklearn.model_selection import train_test_split
X_param = newdf[['R&D Spend', 'Administration', 'Marketing Spend', 'NY', 'FL']]
print(X_param)
y_param = newdf[['Profit']]
print(y_param)

X_train, X_test, y_train, y_test = train_test_split(X_param, y_param, random_state=1)
print('aaa')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# przygotowanie modelu
from sklearn import linear_model
regr = linear_model.LinearRegression()

# trenowanie modelu
regr.fit(X_train, y_train)

# predykcja na danych testowych
y_pred = regr.predict(X_test)

# wyniki

# wspolczynniki
print('wspolczynnik a=', regr.coef_)
print('wspolczynnik wyraz wolny b=', regr.intercept_)

# jaki bład sredniokwadratowy
from sklearn.metrics import mean_squared_error
print('blad sredniokwadratoey', mean_squared_error(y_test, y_pred))

# obliczenie R^2
print('R^2', regr.score(X_train, y_train))




