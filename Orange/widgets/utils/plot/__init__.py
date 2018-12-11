"""
Plot classes and tools that were once used in Orange widgets

Due to lack of maintenance (non-functioning), the majority of it has been
stripped in this commit
"""

# The package is pulling names from modules with defined __all__, except
# for owconstants, where all constants are exported
# pylint: disable=wildcard-import

from .owplotgui import *
from .owpalette import *
from .owconstants import *
