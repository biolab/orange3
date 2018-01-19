import Orange
iris = Orange.data.Table("iris.tab")
disc = Orange.preprocess.Discretize()
disc.method = Orange.preprocess.discretize.EqualFreq(n=3)
d_iris = disc(iris)

print("Original data set:")
for e in iris[:3]:
    print(e)

print("Discretized data set:")
for e in d_iris[:3]:
    print(e)
