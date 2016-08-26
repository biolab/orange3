import Orange

data = Orange.data.Table("voting")
classifier = Orange.classification.LogisticRegressionLearner(data)
target_class = 1
print("Probabilities for %s:" % data.domain.class_var.values[target_class])
probabilities = classifier(data, 1)
for p, (idx, r) in zip(probabilities[5:8], data.iloc[5:8].iterrows()):
    print(p[target_class], r[r.domain.class_var])
