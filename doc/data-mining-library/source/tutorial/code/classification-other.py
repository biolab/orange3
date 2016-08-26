import Orange

data = Orange.data.Table("voting")
test = data.sample(n=5)
train = data.loc[~data.isin(test).all(axis=1)]

tree = Orange.classification.tree.TreeLearner(max_depth=3)
knn = Orange.classification.knn.KNNLearner(n_neighbors=3)
lr = Orange.classification.LogisticRegressionLearner(C=0.1)

learners = [tree, knn, lr]
classifiers = [learner(train) for learner in learners]

target = 0
print("Probabilities for %s:" % data.domain.class_var.values[target])
print("original class ", " ".join("%-5s" % l.name for l in classifiers))

c_values = data.domain.class_var.values
for idx, r in test.iterrows():
    print(("{:<15}" + " {:.3f}"*len(classifiers)).format(
        r[r.domain.class_var],
        *(c(r, 1)[0][target] for c in classifiers)))
