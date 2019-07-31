import pandas
import matplotlib as plt

df = pandas.read_csv('data-geo.csv', sep=',', header=0)
# print(df)
# print(df.head())
# print(df.tail())
print(df.shape)

# kolumna
# print(df['City'])

# 2 kolumny
# print(df[['City', 'State']])

# konkretne zalozenia

# print(df[df['State'] == 'Texas'])

# numer wiersza
# print(df.iloc[120])

# sortowanie wd kolumny
# print(df.sort_values(by=['2015 median sales price'], ascending=False))

# modyfikowanie data frame'a
# df['newcol'] = 'test'
# print(df)

# modyfikowanie df z warunkiem
def addColumn(row):
    if row['2015 median sales price'] > 200:
        return 'high'
    else:
        return 'high'

# apply stasuje do kazdego wiersza a axis 1 oznacza ze do kolumn czyli osi pionowej

# df['isHigh'] = df.apply(addColumn, axis=1)
# print(df)

# group_by po jakich polach ma byc grupowany, zgrupowało po sredniej tam gdzie były liczby
# poczytaj o funkcjach agregujących

dfnew =  df.groupby(['State']).mean()


# rysowanie wykresów normalny i slupkowy
dfnew.plot()
plt.pyplot.show()
dfnew['2015 median sales price'].plot(kind='bar')
plt.pyplot.show()


