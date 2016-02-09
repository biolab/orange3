# Wildcard imports are used to pull all widget tests into this namespace
# pylint: disable=wildcard-import

import unittest

from Orange.widgets.tests.test_setting_provider import *
from Orange.widgets.tests.test_settings_handler import *
from Orange.widgets.tests.test_context_handler import *
from Orange.widgets.tests.test_class_values_context_handler import *
from Orange.widgets.tests.test_domain_context_handler import *
from Orange.widgets.tests.test_perfect_domain_context_handler import *

from Orange.widgets.tests.test_owselectcolumns import *
from Orange.widgets.tests.test_scatterplot_density import *

class Test(unittest.TestCase):
    pass
