from .continuize import *
from .discretize import *
from .fss import *
from .impute import *
from .normalize import *
from .remove import *
from .preprocess import *

from Orange.misc.lazy_module import _LazyModule
transformation = _LazyModule("preprocess.transformation")
score = _LazyModule("preprocess.score")
discretize = _LazyModule("preprocess.discretize")
continuize = _LazyModule("preprocess.continuize")
normalize = _LazyModule("preprocess.normalize")
remove = _LazyModule("preprocess.remove")
