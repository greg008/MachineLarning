import pandas

df = pandas.read_csv('data-geo.csv', sep=',', header=0)
# print(df)
# print(df.head())
# print(df.tail())
print(df.shape)

# kolumna
# print(df['City'])

# 2 kolumny
print(df[['City', 'State']])

# konkretne zalozenia

print(df[df['State'] == 'Texas'])

