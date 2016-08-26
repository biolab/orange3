import Orange

data = Orange.data.Table("voting")
classifier = Orange.classification.LogisticRegressionLearner(data)
c_values = data.domain.class_var.values
for idx, r in data.iloc[5:8].iterrows():
    c = classifier(r)
    print("{}, originally {}".format(c_values[int(classifier(r)[0])], r[r.domain.class_var]))
