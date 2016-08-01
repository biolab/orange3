import Orange

data = Orange.data.Table("housing")
learner = Orange.regression.LinearRegressionLearner()
model = learner(data)

print("predicted, observed:")
for idx, r in data.iloc[:3].iterrows():
    print("%.1f, %.1f" % (model(r)[0], r[r.domain.class_var]))
