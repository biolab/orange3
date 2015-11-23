import Orange

data = Orange.data.Table("titanic")
lr = Orange.classification.LogisticRegressionLearner(data)
# print(Orange.classification.logreg.dump(lr))

tree = Orange.classification.tree.TreeLearner(data)
# print tree.to_string()
