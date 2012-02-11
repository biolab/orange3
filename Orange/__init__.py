import orange

# Definitely ugly, but I see no other workaround.
# When, e.g. data.io executes "from orange import ExampleTable"
# orange gets imported again since it is not in sys.modules
# before this entire file is executed
import sys
sys.modules["orange"] = orange

import warnings

def _import(name):
    try:
        __import__(name, globals(), locals(), [], -1)
    except Exception:
        warnings.warn("Some features are disabled, because Orange could not import: " + name, UserWarning, 2)

_import("misc")
_import("data")
_import("data.io")
_import("data.sample")
_import("data.utils")
_import("data.discretization")
_import("data.filter")
_import("data.imputation")

_import("network")

_import("stat")

_import("statistics")
_import("statistics.estimate")
_import("statistics.contingency")
_import("statistics.distribution")
_import("statistics.basic")
_import("statistics.evd")

_import("classification")
_import("classification.tree")

_import("classification.rules")

_import("classification.lookup")
_import("classification.bayes")
_import("classification.svm")
_import("classification.logreg")
_import("classification.knn")
_import("classification.majority")

_import("optimization")

_import("projection")
_import("projection.linear")
_import("projection.mds")
_import("projection.som")
_import("projection.pca")

_import("ensemble")
_import("ensemble.bagging")
_import("ensemble.boosting")
_import("ensemble.forest")

_import("regression")
_import("regression.base")
_import("regression.earth")
_import("regression.lasso")
_import("regression.linear")
_import("regression.mean")
_import("regression.pls")
_import("regression.tree")

_import("multitarget")
_import("multitarget.tree")

_import("multilabel")
_import("multilabel.multibase")
_import("multilabel.br")
_import("multilabel.lp")
_import("multilabel.mlknn")
_import("multilabel.brknn")
_import("multilabel.mulan")

_import("associate")

_import("preprocess")
_import("preprocess.scaling")

_import("distance")

_import("wrappers")

_import("featureConstruction")
_import("featureConstruction.univariate")
_import("featureConstruction.functionDecomposition")

_import("evaluation")
_import("evaluation.scoring")
_import("evaluation.testing")
_import("evaluation.reliability")

_import("clustering")
_import("clustering.kmeans")
_import("clustering.hierarchical")
_import("clustering.consensus")

_import("misc")
_import("misc.environ")
_import("misc.counters")
_import("misc.addons")
_import("misc.render")
_import("misc.selection")
_import("misc.serverfiles")
#_import("misc.r")
