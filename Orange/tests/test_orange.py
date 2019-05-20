import unittest


class TestOrange(unittest.TestCase):
    def test_orange_has_modules(self):
        import pkgutil
        import Orange
        unimported = ['canvas', 'datasets', 'testing', 'tests', 'setup',
                      'util', 'widgets']
        for _, name, __ in pkgutil.iter_modules(Orange.__path__):
            if name not in unimported:
                self.assertIn(name, Orange.__dict__)
