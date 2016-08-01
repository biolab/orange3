import Orange
iris = Orange.data.Table("iris.tab")
disc = Orange.preprocess.Discretize()
disc.method = Orange.preprocess.discretize.EqualFreq(n=2)
d_disc_iris = disc(iris)
disc_iris = Orange.data.Table(d_disc_iris, iris)

print("Original data set:")
for idx, row in iris.iloc[:3].iterrows():
    print(row)

print("Discretized data set:")
for idx, row in disc_iris.iloc[:3].iterrows():
    print(row)
