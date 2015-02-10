import Orange
iris = Orange.data.Table("iris.tab")
disc_iris = Orange.preprocess.DiscretizeTable(iris,
    method=Orange.preprocess.EqualFreq(n=3))

print("Original data set:")
for e in iris[:3]:
    print(e)

print("Discretized data set:")
for e in disc_iris[:3]:
    print(e)
