import unittest

class TestOrange(unittest.TestCase):
    def test_import_all(self):
        import pkgutil
        import Orange
        Orange.import_all()
        unimported = ['canvas', 'datasets', 'testing', 'tests', 'setup',
                      'widgets']
        for _, name, __ in pkgutil.iter_modules(Orange.__path__):
            if name not in unimported:
                self.assertIn(name, Orange.__dict__)
