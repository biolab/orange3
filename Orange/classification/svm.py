import sklearn.svm as skl_svm

from Orange.classification import SklLearner, SklModel
from Orange.base import SklLearner as SklLearnerBase
from Orange.preprocess import Normalize
from Orange import options

__all__ = ["SVMLearner", "LinearSVMLearner", "NuSVMLearner",
           "OneClassSVMLearner"]


svm_pps = SklLearner.preprocessors + [Normalize()]


class SVMClassifier(SklModel):

    def predict(self, X):
        value = self.skl_model.predict(X)
        if self.skl_model.probability:
            prob = self.skl_model.predict_proba(X)
            return value, prob
        return value


class SVMOptions:
    KERNERLS = (
        options.Choice('linear', label='x⋅y'),
        options.Choice('rbf', related_options=('gamma',), label='exp(-g|x-y|²)'),
        options.Choice('poly', verbose_name='Polynomial',
                       related_options=('degree', 'coef0', 'gamma'),
                       label='(g x⋅y + c)<sup>d</sup>'),
        options.Choice('sigmoid', related_options=('gamma',),
                       label='tanh(g x⋅y + c)'),
    )

    kernels = [
        options.ChoiceOption('kernel', choices=KERNERLS, default='rbf'),
        options.IntegerOption('degree', default=2, range=(1, 10), step=1,
                              verbose_name='d'),
        options.DisableableOption(
            'gamma', disable_value='auto', verbose_name='g',
            option=options.FloatOption(default=.1, range=(0., 1.), step=0.05,),
        ),
        options.FloatOption('coef0', default=0., range=(-100, 100),
                            step=1., verbose_name='c'),
    ]

    C = options.FloatOption('C', verbose_name='Cost (C)', default=1.,
                            range=(.0001, 1000), step=.001)
    nu = options.FloatOption('nu', verbose_name='Complexity (ν)', default=.5,
                             range=(.05, 1.), step=.05)

    tol = options.FloatOption('tol', default=1e-3, range=(1e-5, 1.), step=1e-4)
    max_inter = options.DisableableOption(
        'max_iter', disable_label='No limit', disable_value=-1,
        option=options.IntegerOption(default=1000, range=(10, 10**6), step=10),
    )

    shrinking = options.BoolOption('shrinking', default=True)
    probability = options.BoolOption('probability', default=True)
    cache_size = options.IntegerOption('cache_size', default=200, range=(50, 10 ** 4))


class SVMLearner(SklLearner):
    __wraps__ = skl_svm.SVC
    __returns__ = SVMClassifier
    name = 'svm'
    verbose_name = 'C-SVM'
    preprocessors = svm_pps

    options = SVMOptions.kernels + [
        SVMOptions.C,
        SVMOptions.shrinking,
        SVMOptions.probability,
        SVMOptions.tol,
        SVMOptions.max_inter,
        SVMOptions.cache_size,
    ]

    class GUI:
        base_svm = [
            options.ChoiceGroup('kernel', ('gamma', 'degree', 'coef0')),
            options.OptionGroup('Optimization', ('tol', 'max_iter')),
        ]
        main_scheme = ['C'] + base_svm


class LinearSVMLearner(SklLearner):
    __wraps__ = skl_svm.LinearSVC
    name = 'linear svm'
    verbose_name = 'Linear SVM'
    preprocessors = svm_pps

    options = [
        options.ChoiceOption('penalty', choices=('l1', 'l2'), default='l2'),
        options.ChoiceOption('loss', choices=('hinge', 'squared_hinge'),
                             default='squared_hinge'),
        options.BoolOption('dual', default=True),
        SVMOptions.tol,
        SVMOptions.C,
        options.ObjectOption('multi_class', default='ovr'),
        options.BoolOption('fit_intercept', default=True),
        options.BoolOption('intercept_scaling', default=True),
        options.ObjectOption('random_state', default=None),

    ]


class NuSVMClassifier(SklModel):

    def predict(self, X):
        value = self.skl_model.predict(X)
        if self.skl_model.probability:
            prob = self.skl_model.predict_proba(X)
            return value, prob
        return value


class NuSVMLearner(SklLearner):
    __wraps__ = skl_svm.NuSVC
    __returns__ = NuSVMClassifier
    name = 'nu svm'
    verbose_name = 'ν-SVM'
    preprocessors = svm_pps

    options = SVMOptions.kernels + [
        SVMOptions.nu,
        SVMOptions.shrinking,
        SVMOptions.probability,
        SVMOptions.tol,
        SVMOptions.max_inter,
        SVMOptions.cache_size,
    ]

    class GUI:
        main_scheme = [
            'nu',
        ] + SVMLearner.GUI.base_svm


class OneClassSVMLearner(SklLearnerBase):
    __wraps__ = skl_svm.OneClassSVM
    name = 'one class svm'
    preprocessors = svm_pps

    options = SVMOptions.kernels + [
        SVMOptions.nu,
        SVMOptions.tol,
        SVMOptions.shrinking,
        SVMOptions.cache_size,
        SVMOptions.max_inter,
    ]

    def fit(self, X, Y=None, W=None):
        clf = self.instance
        if W is not None:
            return self.__returns__(clf.fit(X, W.reshape(-1)))
        return self.__returns__(clf.fit(X))


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('iris')
    learners = [SVMLearner(), NuSVMLearner(), LinearSVMLearner()]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.CA(res)):
        print("learner: {}\nCA: {}\n".format(l, ca))
