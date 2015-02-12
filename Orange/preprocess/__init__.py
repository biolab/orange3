from .continuize import *
from .discretize import *
from .fss import *
from .impute import *
from .preprocess import *
from .score import *

from Orange.misc.lazy_module import _LazyModule
transformation = _LazyModule("preprocess.transformation")
