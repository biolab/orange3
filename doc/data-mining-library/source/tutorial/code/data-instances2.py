import Orange

average = lambda x: sum(x)/len(x)

data = Orange.data.Table("iris")
print("%-15s %s" % ("Feature", "Mean"))
for x in data.domain.attributes:
    print("%-15s %.2f" % (x.name, average([r[x] for idx, r in data.iterrows()])))
