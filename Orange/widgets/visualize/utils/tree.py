import sys
import warnings

from Orange.utils import tree

warnings.warn(
    f"{__name__} module was moved. Use {tree.__name__} instead",
    DeprecationWarning,
    stacklevel=2
)
sys.modules[__name__] = tree