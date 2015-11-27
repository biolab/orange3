import Orange
import numpy as np

data = Orange.data.Table("voting")
classifier = Orange.classification.LogisticRegressionLearner(data)
x = np.sum(data.Y != classifier(data))
