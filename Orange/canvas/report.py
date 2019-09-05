import sys
import warnings
import Orange.widgets.report

warnings.warn(
    f"'{__name__}' is deprecated and will be removed in the future.\n"
    "The contents of this package were moved to 'Orange.widgets.report'. "
    "Please update the imports accordingly.",
    FutureWarning, stacklevel=2
)
sys.modules[__name__] = Orange.widgets.report
