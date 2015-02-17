import Orange
iris = Orange.data.Table("iris.tab")
disc = Orange.preprocess.Discretize()
disc.method = Orange.preprocess.discretize.EqualFreq(n=2)
d_disc_iris = disc(iris)
disc_iris = Orange.data.Table(d_disc_iris, iris)

print("Original data set:")
for e in iris[:3]:
    print(e)

print("Discretized data set:")
for e in disc_iris[:3]:
    print(e)
