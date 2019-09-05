"""
Methods for scoring prediction results (CA, AUC, ...).

Examples
--------
>>> import Orange
>>> data = Orange.data.Table('iris')
>>> learner = Orange.classification.LogisticRegressionLearner()
>>> results = Orange.evaluation.TestOnTrainingData(data, [learner])

"""

import math

import numpy as np
import sklearn.metrics as skl_metrics
from sklearn.metrics import confusion_matrix

from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.misc.wrapper_meta import WrapperMeta

__all__ = ["CA", "Precision", "Recall", "F1", "PrecisionRecallFSupport", "AUC",
           "MSE", "RMSE", "MAE", "R2", "compute_CD", "graph_ranks", "LogLoss"]


class ScoreMetaType(WrapperMeta):
    """
    Maintain a registry of non-abstract subclasses and assign the default
    value of `name`.

    The existing meta class Registry cannot be used since a meta class cannot
    have multiple inherited __new__ methods."""
    def __new__(mcs, name, bases, dict_, **kwargs):
        cls = WrapperMeta.__new__(mcs, name, bases, dict_)
        # Essentially `if cls is not Score`, except that Score may not exist yet
        if hasattr(cls, "registry"):
            if not kwargs.get("abstract"):
                # Don't use inherited names, look into dict_
                cls.name = dict_.get("name", name)
                cls.registry[name] = cls
        else:
            cls.registry = {}
        return cls

    def __init__(cls, *args, **_):
        WrapperMeta.__init__(cls, *args)


class Score(metaclass=ScoreMetaType):
    """
    ${sklpar}
    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.
    """
    __wraps__ = None

    separate_folds = False
    is_scalar = True
    is_binary = False  #: If true, compute_score accepts `target` and `average`
    #: If the class doesn't explicitly contain `abstract=True`, it is not
    #: abstract; essentially, this attribute is not inherited
    abstract = True
    class_types = ()
    name = None
    long_name = None  #: A short user-readable name (e.g. a few words)

    def __new__(cls, results=None, **kwargs):
        self = super().__new__(cls)
        if results is not None:
            self.__init__()
            return self(results, **kwargs)
        else:
            return self

    def __call__(self, results, **kwargs):
        if self.separate_folds and results.score_by_folds and results.folds:
            scores = self.scores_by_folds(results, **kwargs)
            return self.average(scores)

        return self.compute_score(results, **kwargs)

    def average(self, scores):
        if self.is_scalar:
            return np.mean(scores, axis=0)
        return NotImplementedError

    def scores_by_folds(self, results, **kwargs):
        nfolds = len(results.folds)
        nmodels = len(results.predicted)
        if self.is_scalar:
            scores = np.empty((nfolds, nmodels), dtype=np.float64)
        else:
            scores = [None] * nfolds
        for fold in range(nfolds):
            fold_results = results.get_fold(fold)
            scores[fold] = self.compute_score(fold_results, **kwargs)
        return scores

    def compute_score(self, results):
        wraps = type(self).__wraps__  # self.__wraps__ is invisible
        if wraps:
            return self.from_predicted(results, wraps)
        else:
            return NotImplementedError

    @staticmethod
    def from_predicted(results, score_function, **kwargs):
        return np.fromiter(
            (score_function(results.actual, predicted, **kwargs)
             for predicted in results.predicted),
            dtype=np.float64, count=len(results.predicted))


class ClassificationScore(Score, abstract=True):
    class_types = (DiscreteVariable, )


class RegressionScore(Score, abstract=True):
    class_types = (ContinuousVariable, )


# pylint: disable=invalid-name
class CA(ClassificationScore):
    __wraps__ = skl_metrics.accuracy_score
    long_name = "Classification accuracy"


class PrecisionRecallFSupport(ClassificationScore):
    __wraps__ = skl_metrics.precision_recall_fscore_support
    is_scalar = False


class TargetScore(ClassificationScore):
    """
    Base class for scorers that need a target value (a "positive" class).

    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.

    target : int, optional (default=None)
        Target class value.
        When None:
          - if averaging is specified, use all classes and average results
          - if average is 'binary' and class variable has exactly 2 values,
            use the value '1' as the positive class

    average: str, method for averaging (default='binary')
        Default requires a binary class or target to be set.
        Options: 'weighted', 'macro', 'micro', None

    """
    is_binary = True
    abstract = True
    __wraps__ = None  # Subclasses should set the scoring function

    def compute_score(self, results, target=None, average='binary'):
        if average == 'binary':
            if target is None:
                if len(results.domain.class_var.values) > 2:
                    raise ValueError(
                        "Multiclass data: specify target class or select "
                        "averaging ('weighted', 'macro', 'micro')")
                target = 1  # Default: use 1 as "positive" class
            average = None
        labels = None if target is None else [target]
        return self.from_predicted(
            results, type(self).__wraps__, labels=labels, average=average)


class Precision(TargetScore):
    __wraps__ = skl_metrics.precision_score


class Recall(TargetScore):
    __wraps__ = skl_metrics.recall_score


class F1(TargetScore):
    __wraps__ = skl_metrics.f1_score


class AUC(ClassificationScore):
    """
    ${sklpar}

    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.

    target : int, optional (default=None)
        Value of class to report.
    """
    __wraps__ = skl_metrics.roc_auc_score
    separate_folds = True
    is_binary = True
    long_name = "Area under ROC curve"

    @staticmethod
    def calculate_weights(results):
        classes = np.unique(results.actual)
        class_cases = [sum(results.actual == class_)
                       for class_ in classes]
        N = results.actual.shape[0]
        weights = np.array([c * (N - c) for c in class_cases])
        wsum = np.sum(weights)
        if wsum == 0:
            raise ValueError("Class variable has less than two values")
        else:
            return weights / wsum

    @staticmethod
    def single_class_auc(results, target):
        y = np.array(results.actual == target, dtype=int)
        return np.fromiter(
            (skl_metrics.roc_auc_score(y, probabilities[:, int(target)])
             for probabilities in results.probabilities),
            dtype=np.float64, count=len(results.predicted))

    def multi_class_auc(self, results):
        classes = np.unique(results.actual)
        weights = self.calculate_weights(results)
        auc_array = np.array([self.single_class_auc(results, class_)
                              for class_ in classes])
        return np.sum(auc_array.T * weights, axis=1)

    def compute_score(self, results, target=None, average=None):
        domain = results.domain
        n_classes = len(domain.class_var.values)

        if n_classes < 2:
            raise ValueError("Class variable has less than two values")
        elif n_classes == 2:
            return self.single_class_auc(results, 1)
        else:
            if target is None:
                return self.multi_class_auc(results)
            else:
                return self.single_class_auc(results, target)


class LogLoss(ClassificationScore):
    """
    ${sklpar}

    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.

    eps : float
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).

    normalize : bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Examples
    --------
    >>> Orange.evaluation.LogLoss(results)
    array([ 0.3...])

    """
    __wraps__ = skl_metrics.log_loss

    def compute_score(self, results, eps=1e-15, normalize=True,
                      sample_weight=None):
        return np.fromiter(
            (skl_metrics.log_loss(results.actual,
                                  probabilities,
                                  eps=eps,
                                  normalize=normalize,
                                  sample_weight=sample_weight)
             for probabilities in results.probabilities),
            dtype=np.float64, count=len(results.probabilities))


class Specificity(ClassificationScore):
    is_binary = True

    @staticmethod
    def calculate_weights(results):
        classes, counts = np.unique(results.actual, return_counts=True)
        n = np.array(results.actual).shape[0]
        return counts / n, classes

    @staticmethod
    def specificity(y_true, y_pred):
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def single_class_specificity(self, results, target):
        y_true = (np.array(results.actual) == target).astype(int)
        return np.fromiter(
            (self.specificity(y_true,
                              np.array(predicted == target, dtype=int))
             for predicted in results.predicted),
            dtype=np.float64, count=len(results.predicted))

    def multi_class_specificity(self, results):
        weights, classes = self.calculate_weights(results)
        scores = np.array([self.single_class_specificity(results, class_)
                           for class_ in classes])
        return np.sum(scores.T * weights, axis=1)

    def compute_score(self, results, target=None, average="binary"):
        domain = results.domain
        n_classes = len(domain.class_var.values)

        if target is None:
            if average == "weighted":
                return self.multi_class_specificity(results)
            elif average == "binary":  # average is binary
                if n_classes != 2:
                    raise ValueError(
                        "Binary averaging needs two classes in data: "
                        "specify target class or use "
                        "weighted averaging.")
                return self.single_class_specificity(results, 1)
            else:
                raise ValueError(
                    "Wrong parameters: For averaging select one of the "
                    "following values: ('weighted', 'binary')")
        elif target is not None:
            return self.single_class_specificity(results, target)

# Regression scores


class MSE(RegressionScore):
    __wraps__ = skl_metrics.mean_squared_error
    long_name = "Mean square error"


class RMSE(RegressionScore):
    long_name = "Root mean square error"

    def compute_score(self, results):
        return np.sqrt(MSE(results))


class MAE(RegressionScore):
    __wraps__ = skl_metrics.mean_absolute_error
    long_name = "Mean absolute error"


# pylint: disable=invalid-name
class R2(RegressionScore):
    __wraps__ = skl_metrics.r2_score
    long_name = "Coefficient of determination"


class CVRMSE(RegressionScore):
    long_name = "Coefficient of variation of the RMSE"

    def compute_score(self, results):
        mean = np.nanmean(results.actual)
        if mean < 1e-10:
            raise ValueError("Mean value is too small")
        return RMSE(results) / mean * 100


# CD scores and plot

def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd


def graph_ranks(avranks, names, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd and cdmethod is None:
        # get pairs of non significant methods

        def get_lines(sums, hsd):
            # get all pairs
            lsums = len(sums)
            allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
            # remove not significant
            notSig = [(i, j) for i, j in allpairs
                      if abs(sums[i] - sums[j]) <= hsd]
            # keep only longest

            def no_longer(ij_tuple, notSig):
                i, j = ij_tuple
                for i1, j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

            return longest

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]


    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center")

    if cd and cdmethod is None:
        # upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick / 2),
              (begin, distanceh - bigtick / 2)],
             linewidth=0.7)
        line([(end, distanceh + bigtick / 2),
              (end, distanceh - bigtick / 2)],
             linewidth=0.7)
        text((begin + end) / 2, distanceh - 0.05, "CD",
             ha="center", va="bottom")

        # no-significance lines
        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l, r in lines:
                line([(rankpos(ssums[l]) - side, start),
                      (rankpos(ssums[r]) + side, start)],
                     linewidth=2.5)
                start += height

        draw_lines(lines)

    elif cd:
        begin = rankpos(avranks[cdmethod] - cd)
        end = rankpos(avranks[cdmethod] + cd)
        line([(begin, cline), (end, cline)],
             linewidth=2.5)
        line([(begin, cline + bigtick / 2),
              (begin, cline - bigtick / 2)],
             linewidth=2.5)
        line([(end, cline + bigtick / 2),
              (end, cline - bigtick / 2)],
             linewidth=2.5)

    if filename:
        print_figure(fig, filename, **kwargs)
