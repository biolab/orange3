import Orange
import random

random.seed(42)
data = Orange.data.Table("housing")
test = Orange.data.Table(data.domain, random.sample(data, 5))
train = Orange.data.Table(data.domain, [d for d in data if d not in test])

lin = Orange.regression.linear.LinearRegressionLearner()
rf = Orange.regression.random_forest.RandomForestRegressionLearner()
rf.name = "rf"
ridge = Orange.regression.RidgeRegressionLearner()

learners = [lin, rf, ridge]
regressors = [learner(train) for learner in learners]

print("y   ", " ".join("%5s" % l.name for l in regressors))

for d in test:
    print(("{:<5}" + " {:5.1f}"*len(regressors)).format(
        d.get_class(),
        *(r(d)[0] for r in regressors)))
