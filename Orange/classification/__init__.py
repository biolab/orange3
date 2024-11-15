# Pull members from modules to Orange.classification namespace
# pylint: disable=wildcard-import

from .base_classification import (ModelClassification as Model,
                                  LearnerClassification as Learner,
                                  SklModelClassification as SklModel,
                                  SklLearnerClassification as SklLearner)
from .knn import *
from .logistic_regression import *
from .majority import *
from .naive_bayes import *
from .random_forest import *
from .softmax_regression import *
from .svm import *
from .tree import *
from .simple_tree import *
from .simple_random_forest import *
from .outlier_detection import *
from .rules import *
from .sgd import *
from .neural_network import *
from .calibration import *
from .scoringsheet import *
try:
    from .catgb import *
except ModuleNotFoundError:
    pass
from .gb import *
try:
    from .xgb import *
except Exception:
    pass
