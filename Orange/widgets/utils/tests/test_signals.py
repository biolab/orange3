import unittest
from unittest.mock import Mock

from Orange.widgets.utils.signals import lazy_table_transform


class TestSignals(unittest.TestCase):
    def test_lazy_table_transform(self):
        data = Mock()
        data.__len__ = lambda _: 42
        data.transform = Mock()

        domain = Mock()

        lazy_trans = lazy_table_transform(domain, data)

        data.transform.assert_not_called()
        self.assertEqual(lazy_trans.length, 42)
        self.assertIs(lazy_trans.domain, domain)
        self.assertFalse(lazy_trans.is_cached)

        self.assertIs(lazy_trans.get_value(), data.transform.return_value)
        data.transform.assert_called_once()


if __name__ == '__main__':
    unittest.main()
