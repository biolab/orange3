import Orange

data = Orange.data.Table("voting")
classifier = Orange.classification.LogisticRegressionLearner(data)
c_values = data.domain.class_var.values
for d in data[5:8]:
    c = classifier(d)
    print("{}, originally {}".format(c_values[int(classifier(d)[0])],
                                     d.get_class()))
