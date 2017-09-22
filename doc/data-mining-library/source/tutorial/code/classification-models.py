import Orange

data = Orange.data.Table("titanic")
lr = Orange.classification.LogisticRegressionLearner(data)

tree = Orange.classification.tree.TreeLearner(data)
