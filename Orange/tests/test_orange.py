import unittest
import warnings
import os

import scipy.sparse as sp


class TestOrange(unittest.TestCase):
    def test_orange_has_modules(self):
        import pkgutil
        import Orange
        unimported = ['canvas', 'datasets', 'testing', 'tests', 'setup',
                      'util', 'widgets']
        for _, name, __ in pkgutil.iter_modules(Orange.__path__):
            if name not in unimported:
                self.assertIn(name, Orange.__dict__)

    @unittest.skipUnless(
        os.environ.get("TRAVIS"),
        "Travis has latest versions; Appveyor doesn't, and users don't care")
    def test_remove_matrix_deprecation_filter(self):
        # Override filter in Orange.__init__
        warnings.filterwarnings(
            "once", ".*the matrix subclass.*", PendingDeprecationWarning)
        with self.assertWarns(
                PendingDeprecationWarning,
                msg="Remove filter for PendingDeprecationWarning of np.matrix "
                    "from Orange.__init__"):
            sp.lil_matrix([1, 2, 3])
