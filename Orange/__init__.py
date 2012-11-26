from __future__ import absolute_import
from importlib import import_module

try:
    from .import version
    # Always use short_version here (see PEP 386)
    __version__ = version.short_version
    __hg_revision__ = version.hg_revision
except ImportError:
    __version__ = "unknown"
    __hg_revision__ = "unknown"

ADDONS_ENTRY_POINT = 'orange.addons'

import warnings
import pkg_resources

alreadyWarned = False
disabledMsg = "Some features will be disabled due to failing modules\n"


def _import(name):
    global alreadyWarned
    try:
        import_module(name, package='Orange')
    except ImportError as err:
        warnings.warn("%sImporting '%s' failed: %s" %
                      (disabledMsg if not alreadyWarned else "", name, err),
                      UserWarning, 2)
        alreadyWarned = True


def _import_addons():
    globals_dict = globals()
    for entry_point in pkg_resources.iter_entry_points(ADDONS_ENTRY_POINT):
        try:
            module = entry_point.load()
            # Dot is not allowed in an entry point name (it should
            # be a Python identifier, because it is used as a class
            # name), so we are using __ instead
            name = entry_point.name.replace('__', '.')
            if '.' not in name:
                globals_dict[name] = module
            else:
                path, mod = name.rsplit('.', 1)
                parent_module_name = 'Orange.%s' % (path,)
                try:
                    parent_module = sys.modules[parent_module_name]
                except KeyError:
                    warnings.warn(
                        "Loading add-on '%s' failed because destination namespace point '%s' was not found." % (
                        entry_point.name, parent_module_name), UserWarning, 2)
                    continue
                setattr(parent_module, mod, module)
            sys.modules['Orange.%s' % (name,)] = module
        except ImportError as err:
            warnings.warn("Importing add-on '%s' failed: %s" % (entry_point.name, err), UserWarning, 2)
        except pkg_resources.DistributionNotFound as err:
            warnings.warn("Loading add-on '%s' failed because of a missing dependency: '%s'" % (entry_point.name, err),
                          UserWarning, 2)
        except Exception as err:
            warnings.warn("An exception occurred during the loading of '%s':\n%r" % (entry_point.name, err),
                          UserWarning, 2)


#_import("utils")

_import(".data")
_import(".data.io")
_import(".data.table")
_import(".data.value")
_import(".data.instance")
_import(".data.variable")
_import(".data.domain")

"""
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

_import("tuning")

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

_import("clustering")
_import("clustering.kmeans")
_import("clustering.hierarchical")
_import("clustering.consensus")

_import("misc")

_import("utils") #TODO hide utils from the user
_import("utils.environ")
_import("utils.counters")
_import("utils.addons")
_import("utils.render")
_import("utils.serverfiles")

_import_addons()
"""

del _import
del _import_addons
del alreadyWarned
del disabledMsg
