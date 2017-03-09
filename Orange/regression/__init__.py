# Pull members from modules to Orange.regression namespace
# pylint: disable=wildcard-import

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
from Orange.classification.simple_tree import *
from .neural_network import *
