import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Ładowanie danych
iris = pandas.read_csv('iris.csv', sep=',', header=None)

print(iris)

# przygotowanie danych
# usuwanie wierszy bez atrybutu

iris = iris.dropna()

# podział danych na atrybuty i klase

x_iris = iris[[0, 1, 2, 3]]
y_iris = iris[4]

# podział na zbiór testowy i treningowy
# random_state=1 domuslnie 75% na 25%

X_train, X_test, Y_train, Y_test = train_test_split(x_iris, y_iris, random_state=1)

# przygotowanie modelu

model = svm.SVC(gamma=0.001, C=100)

# trenowanie modelu

my_model = model.fit(X_train, Y_train)
print(my_model)
# predykcja na danych testowych

result = my_model.predict(X_test)
accur = accuracy_score(Y_test, result)
print(accur)
