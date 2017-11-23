import sys
import warnings
import Orange.widgets.report

warnings.warn(
    "'Orange.canvas.report' was moved to 'Orange.widgets.report'",
    DeprecationWarning,
    stacklevel=2
)
sys.modules[__name__] = Orange.widgets.report
