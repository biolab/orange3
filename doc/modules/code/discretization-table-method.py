import Orange
iris = Orange.data.Table("iris.tab")
disc = Orange.preprocess.DiscretizeTable()
disc.method = Orange.preprocess.EqualFreq(n=2)
disc_iris = disc(iris)

print("Original data set:")
for e in iris[:3]:
    print(e)

print("Discretized data set:")
for e in disc_iris[:3]:
    print(e)
