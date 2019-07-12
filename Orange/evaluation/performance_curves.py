import numpy as np


class Curves:
    # names of scores are standard acronyms, pylint: disable=invalid-name
    """
    Computation of performance curves (ca, f1, precision, recall and the rest
    of the zoo) from test results.

    The class works with binary classes. Attribute `probs` contains ordered
    probabilities and all curves represent performance statistics if an
    instance is classified as positive if it equals or exceeds the threshold
    in `probs`, that is, `sensitivity[i]` is the sensitivity of the classifier
    that classifies an instances as positive if the probability of being
    positive is at least `probs[i]`.

    Class can be constructed by giving `probs` and `ytrue`, or from test
    results (see :obj:`Curves.from_results`). The latter removes instances
    with missing class values or predicted probabilities.

    The class treats all results as obtained from a single run instead of
    computing separate curves and fancy averaging.

    Arguments:
        probs (np.ndarray): vector of predicted probabilities
        ytrue (np.ndarray): corresponding true classes

    Attributes:
        probs (np.ndarray): ordered vector of predicted probabilities
        ytrue (np.ndarray): corresponding true classes
        tot (int): total number of data instances
        p (int): number of real positive instances
        n (int): number of real negative instances
        tp (np.ndarray): number of true positives (property computed from `tn`)
        fp (np.ndarray): number of false positives (property computed from `tn`)
        tn (np.ndarray): number of true negatives (property computed from `tn`)
        fn (np.ndarray): number of false negatives (precomputed, not a property)
    """
    def __init__(self, ytrue, probs):
        sortind = np.argsort(probs)
        self.probs = np.hstack((probs[sortind], [1]))
        self.ytrue = ytrue[sortind]
        self.fn = np.hstack(([0], np.cumsum(self.ytrue)))
        self.tot = len(probs)
        self.p = self.fn[-1]
        self.n = self.tot - self.p

    @classmethod
    def from_results(cls, results, target_class=None, model_index=None):
        """
        Construct an instance of `Curves` from test results.

        Args:
            results (:obj:`Orange.evaluation.testing.Results`): test results
            target_class (int): target class index; if the class is binary,
                this defaults to `1`, otherwise it must be given
            model_index (int): model index; if there is only one model, this
                argument can be omitted

        Returns:
            curves (:obj:`Curves`)
        """
        if model_index is None:
            if results.probabilities.shape[0] != 1:
                raise ValueError("Argument 'model_index' is required when "
                                 "there are multiple models")
            model_index = 0
        if target_class is None:
            if results.probabilities.shape[2] != 2:
                raise ValueError("Argument 'target_class' is required when the "
                                 "class is not binary")
            target_class = 1
        actual = results.actual
        probs = results.probabilities[model_index, :, target_class]
        nans = np.isnan(actual) + np.isnan(probs)
        if nans.any():
            actual = actual[~nans]
            probs = probs[~nans]
        return cls(actual == target_class, probs)

    @property
    def tn(self):
        return np.arange(self.tot + 1) - self.fn

    @property
    def tp(self):
        return self.p - self.fn

    @property
    def fp(self):
        return self.n - self.tn

    def ca(self):
        """Classification accuracy curve"""
        return (self.tp + self.tn) / self.tot

    def f1(self):
        """F1 curve"""
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def sensitivity(self):
        """Sensitivity curve"""
        return self.tp / self.p

    def specificity(self):
        """Specificity curve"""
        return self.tn / self.n

    def precision(self):
        """
        Precision curve

        The last element represents precision at threshold 1. Unless such
        a probability appears in the data, the precision at this point is
        undefined. To avoid this, we copy the previous value to the last.
        """
        tp_fp = np.arange(self.tot, -1, -1)
        tp_fp[-1] = 1  # avoid division by zero
        prec = self.tp / tp_fp
        prec[-1] = prec[-2]
        return prec

    def recall(self):
        """Recall curve"""
        return self.sensitivity()

    def ppv(self):
        """PPV curve; see the comment at :obj:`precision`"""
        return self.precision()

    def npv(self):
        """
        NPV curve

        The first value is undefined (no negative instances). To avoid this,
        we copy the second value into the first.
        """
        tn_fn = np.arange(self.tot + 1)
        tn_fn[0] = 1  # avoid division by zero
        npv = self.tn / tn_fn
        npv[0] = npv[1]
        return npv

    def fpr(self):
        """FPR curve"""
        return self.fp / self.n

    def tpr(self):
        """TPR curve"""
        return self.sensitivity()
