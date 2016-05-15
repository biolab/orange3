import sklearn.svm as skl_svm

from Orange.classification.svm import SVMLearner, NuSVMLearner, SVMOptions
from Orange.regression import SklLearner
from Orange.preprocess import Normalize

__all__ = ["SVRLearner", "NuSVRLearner"]

svm_pps = SklLearner.preprocessors + [Normalize()]


class SVRLearner(SklLearner):
    __wraps__ = skl_svm.SVR
    name = 'svr'
    verbose_name = 'C-SVR'
    preprocessors = svm_pps

    options = SVMOptions.kernels + [
        SVMOptions.C,
        SVMOptions.shrinking,
        SVMOptions.tol,
        SVMOptions.cache_size,
        SVMOptions.max_inter,
    ]

    GUI = SVMLearner.GUI


class NuSVRLearner(SklLearner):
    __wraps__ = skl_svm.NuSVR
    name = 'nu svr'
    verbose_name = 'ν-SVR'
    preprocessors = svm_pps

    options = SVMOptions.kernels + [
        SVMOptions.nu,
        SVMOptions.shrinking,
        SVMOptions.tol,
        SVMOptions.cache_size,
        SVMOptions.max_inter,
    ]
    GUI = NuSVMLearner.GUI


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('housing')
    learners = [SVRLearner(), NuSVRLearner()]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.RMSE(res)):
        print("learner: {}\nRMSE: {}\n".format(l, ca))

