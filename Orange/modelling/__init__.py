# Pull members from modules to Orange.modelling namespace
# pylint: disable=wildcard-import

from .base import *

from .ada_boost import *
from .constant import *
from .knn import *
from .linear import *
from .neural_network import *
from .randomforest import *
from .svm import *
from .tree import *
try:
    from .catgb import *
except ImportError:
    pass
from .gb import *
try:
    from .xgb import *
except ImportError:
    pass
