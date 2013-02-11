"""

*************************
Plot classes and tools for use in Orange widgets
*************************

The main class of this module is :obj:`.OWPlot`, from which all plots 
in visualization widgets should inherit. 

This module also contains plot elements, which are normally used by the :obj:`.OWPlot`, 
but can also be used directly or subclassed

"""

from .owcurve import *
from .owpoint import *
from .owlegend import *
from .owaxis import *
from .owplot import *
from .owtools import *
