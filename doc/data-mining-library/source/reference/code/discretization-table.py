import Orange
iris = Orange.data.Table("iris.tab")
disc = Orange.preprocess.Discretize()
disc.method = Orange.preprocess.discretize.EqualFreq(n=3)
d_iris = disc(iris)

print("Original data set:")
for idx, row in iris.iloc[:3].iterrows():
    print(row)

print("Discretized data set:")
for idx, row in d_iris.iloc[:3].iterrows():
    print(row)
