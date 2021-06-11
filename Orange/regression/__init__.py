# Pull members from modules to Orange.regression namespace
# pylint: disable=wildcard-import,broad-except

from .base_regression import (ModelRegression as Model,
                              LearnerRegression as Learner,
                              SklModelRegression as SklModel,
                              SklLearnerRegression as SklLearner)
from .linear import *
from .mean import *
from .knn import *
from .simple_random_forest import *
from .svm import *
from .random_forest import *
from .tree import *
from .neural_network import *
from ..classification.simple_tree import *
try:
    from .catgb import *
except ModuleNotFoundError:
    pass
from .gb import *
try:
    from .xgb import *
except Exception:
    pass
from .curvefit import *
