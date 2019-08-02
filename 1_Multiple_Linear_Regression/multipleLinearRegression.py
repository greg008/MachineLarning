import pandas
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# wczytanie danych
df = pandas.read_csv('diamonds_multi.csv', sep=';', header=0)
print(df.head())
#pokazuje wartości unikalne z tej kolumny
print(df['color'].unique())
print(df['cut'].unique())

#przygotowanie danych
colorMap = {
    'D': 7,
    'E': 6,
    'F': 5,
    'G': 4,
    'H': 3,
    'I': 2,
    'J': 1,
}

cutMap = {
    'Ideal': 5,
    'Premium': 4,
    'Very Good': 3,
    'Good': 2,
    'Fair': 1,
}
# mapujemy kolor z nieliczbowymi wartosciami na liczbowe
df['colorNum'] = df['color'].map(colorMap)
df['cutNum'] = df['cut'].map(cutMap)
newdf = df.drop(['color', 'cut'], axis=1)

#podział na dane traningowe i testowe
#podział na kolumny do predykcji i do uczenia modelu
X_diam = newdf[['carat', 'x', 'y', 'z', 'colorNum', 'cutNum']]
y_diam = newdf[['price']]

X_train, X_test, y_train, y_test = train_test_split(X_diam, y_diam, random_state=1)
print('aaa')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# przygotowanie modelu
regr = linear_model.LinearRegression()

#trenowanie modelu
regr.fit(X_train, y_train)

# predykcja na danych testowych
y_pred = regr.predict(X_test)

print('===Wyniki===')
# wyniki
# y = a1x1 + a2x2 + a3x3 +... + b
# jakie sa wspolczynniki i wyraz wolny a i b ze wzoru y = ax + b
print('wspolczynnik a=', regr.coef_)
print('wspolczynnik wyraz wolny b=', regr.intercept_)

# jaki bład sredniokwadratowy
print('blad sredniokwadratoey', mean_squared_error(y_test, y_pred))

# wspólczynnik determinacji czyli R^2
print('R^2', regr.score(X_train, y_train))






