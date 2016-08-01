import Orange

data = Orange.data.Table("voting")
test = data.sample(n=5)
train = data.loc[~data.isin(test).all(axis=1)]

lin = Orange.regression.linear.LinearRegressionLearner()
rf = Orange.regression.random_forest.RandomForestRegressionLearner()
rf.name = "rf"
ridge = Orange.regression.RidgeRegressionLearner()

learners = [lin, rf, ridge]
regressors = [learner(train) for learner in learners]

print("y   ", " ".join("%5s" % l.name for l in regressors))

for idx, r in test.iterrows():
    print(("{:<5}" + " {:5.1f}"*len(regressors)).format(
        r[r.domain.class_var],
        *(reg(r)[0] for reg in regressors)))
