from __future__ import absolute_import

__version__ = "2.5a4"

from . import orange

# Definitely ugly, but I see no other workaround.
# When, e.g. data.io executes "from orange import ExampleTable"
# orange gets imported again since it is not in sys.modules
# before this entire file is executed
import sys
sys.modules["orange"] = orange

import warnings

alreadyWarned = False
disabledMsg = "Some features will be disabled due to failing modules\n"
def _import(name):
    global alreadyWarned
    try:
        __import__(name, globals(), locals(), [], -1)
    except ImportError, err:
        warnings.warn("%sImporting '%s' failed: %s" %
            (disabledMsg if not alreadyWarned else "", name, err),
            UserWarning, 2)
        alreadyWarned = True

_import("misc")
_import("data")
_import("data.io")
_import("data.sample")
_import("data.outliers")
_import("data.preprocess")
_import("data.preprocess.scaling")
_import("data.utils")
_import("data.discretization")
_import("data.continuization")
_import("data.filter")
_import("data.imputation")

_import("feature")
_import("feature.construction")
_import("feature.construction.functionDecomposition")
_import("feature.construction.univariate")
_import("feature.discretization")
_import("feature.imputation")
_import("feature.scoring")
_import("feature.selection")

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

_import("ensemble")
_import("ensemble.bagging")
_import("ensemble.boosting")
_import("ensemble.forest")
_import("ensemble.stacking")

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
_import("misc.selection")
#_import("misc.r")

_import("utils") #TODO hide utils from the user
_import("utils.environ")
_import("utils.counters")
_import("utils.addons")
_import("utils.render")
_import("utils.serverfiles")
#
try:
    from . import version
    # Always use short_version here (see PEP 386)
    __version__ = version.short_version
    __hg_revision__ = version.hg_revision
except ImportError:
    # Leave the default version defined at the top.
    pass

