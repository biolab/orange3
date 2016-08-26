import Orange

data = Orange.data.Table("iris")
print("First three data instances:")
for idx, row in data[:3].iterrows():
    print(row)

print("25-th data instance:")
print(data.iloc[24])

name = "sepal width"
print("Value of '%s' for the first instance:" % name, data.loc[data.index[0], name])
print("The 3rd value of the 25th data instance:", data.iloc[24, 2])
