import numpy as np
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples

from Orange.data import Table
from Orange.evaluation.testing import Results
from Orange.evaluation.scoring import Score


__all__ = ['ClusteringEvaluation']


class ClusteringResults(Results):
    def __init__(self, store_data=True, **kwargs):
        super().__init__(store_data=True, **kwargs)

    def get_fold(self, fold):
        results = ClusteringResults()
        results.data = self.data

        if self.folds is None:
            raise ValueError("This 'Results' instance does not have folds.")

        if self.models is not None:
            results.models = self.models[fold]

        results.row_indices = self.row_indices
        results.actual = self.actual
        results.predicted = self.predicted[:, fold, :]
        results.domain = self.domain
        return results


class ClusteringScore(Score):
    considers_actual = False

    def from_predicted(self, results, score_function):
        # Clustering scores from labels
        if self.considers_actual:
            return np.fromiter(
                (score_function(results.actual.flatten(), predicted.flatten())
                 for predicted in results.predicted),
                dtype=np.float64, count=len(results.predicted))
        # Clustering scores from data only
        else:
            return np.fromiter(
                (score_function(results.data.X, predicted.flatten())
                 for predicted in results.predicted),
                dtype=np.float64, count=len(results.predicted))


class Silhouette(ClusteringScore):
    separate_folds = True

    def compute_score(self, results):
        return self.from_predicted(results, silhouette_score)


class AdjustedMutualInfoScore(ClusteringScore):
    separate_folds = True
    considers_actual = True

    def compute_score(self, results):
        return self.from_predicted(results, adjusted_mutual_info_score)


class ClusteringEvaluation(ClusteringResults):
    """
    Clustering evaluation.

    If the constructor is given the data and a list of learning algorithms, it
    runs clustering and returns an instance of `Results` containing the
    predicted clustering labels.

    .. attribute:: k
        The number of runs.

    """
    def __init__(self, data, learners, k=1,
                 store_models=False):
        super().__init__(data=data, nmethods=len(learners), store_data=True,
                         store_models=store_models, predicted=None)

        self.k = k
        Y = data.Y.copy().flatten()

        self.predicted = np.empty((len(learners), self.k, len(data)))
        self.folds = range(k)
        self.row_indices = np.arange(len(data))
        self.actual = data.Y.flatten() if hasattr(data, "Y") else None

        if self.store_models:
            self.models = []

        for k in range(self.k):

            if self.store_models:
                fold_models = []
                self.models.append(fold_models)

            for i, learner in enumerate(learners):
                model = learner(data)
                if self.store_models:
                    fold_models.append(model)

                labels = model(data)
                self.predicted[i, k, :] = labels.X.flatten()



def graph_silhouette(X, y, xlim=None, colors=None, figsize=None, filename=None):
    """
    Silhouette plot.
    :param filename:
        Output file name.
    :param X Orange.data.Table or numpy.ndarray
        Data table.
    :param y Orange.data.Table or numpy.ndarray:
        Cluster labels (integers).
    :param colors list, optional (default = None):
            List of colors. If provided, it must equal the number of clusters.
    :param figsize tuple (float, float):
            Figure size (width, height) in inches.
    :param xlim tuple (float, float):
            Limit x-axis values.
    """
    # If the module is not there, let the user install it
    # pylint: disable=import-error
    import matplotlib.pyplot as plt

    if isinstance(X, Table):
        X = X.X
    if isinstance(y, Table):
        y = y.X
    y = y.ravel()

    # Detect number of clusters and set colors
    N = len(set(y))
    if isinstance(colors, type(None)):
        colors = ["g" if i % 2 else "b" for i in range(N)]
    elif len(colors) != N:
        import sys
        sys.stderr.write("Number of colors does not match the number of clusters. \n")
        return

    # Silhouette coefficients
    s = silhouette_samples(X, y)
    s = s[np.argsort(y)]  # Sort by clusters
    parts = []
    # Within clusters sort by silhouette scores
    for label, (i, j) in enumerate([(sum(y == c1), sum(y == c1) + sum(y == c2))
                                    for c1, c2 in zip(range(-1, N-1), range(0, N))]):
        scores = sorted(s[i:j])
        parts.append((scores, label))

    # Plot data
    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.title("Silhouette score")
    total = 0
    centers = []
    for i, (scores, label) in enumerate(parts):
        plt.barh(range(total, total + len(scores)),
                 scores, color=colors[i], edgecolor=colors[i])
        centers.append(total+len(scores)/2)
        total += len(scores)
    if not isinstance(xlim, type(None)):
        plt.xlim(xlim)
    plt.yticks(centers)
    plt.gca().set_yticklabels(range(N))
    plt.ylabel("Cluster label")
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
