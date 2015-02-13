import unittest
from mock import Mock, MagicMock, patch

import Orange


class TestPreprocess(unittest.TestCase):
    def test_read_data_calls_reader(self):
        class MockPreprocessor(Orange.preprocess.preprocess.Preprocess):
            __init__ = Mock(return_value=None)
            __call__ = Mock()
            @classmethod
            def reset(cls):
                cls.__init__.reset_mock()
                cls.__call__.reset_mock()

        table = Mock(Orange.data.Table)
        MockPreprocessor(table, 1, 2, a=3)
        MockPreprocessor.__init__.assert_called_with(1, 2, a=3)
        MockPreprocessor.__call__.assert_called_with(table)
        MockPreprocessor.reset()

        MockPreprocessor(1, 2, a=3)
        MockPreprocessor.__init__.assert_called_with(1, 2, a=3)
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)

        MockPreprocessor(a=3)
        MockPreprocessor.__init__.assert_called_with(a=3)
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)

        MockPreprocessor()
        MockPreprocessor.__init__.assert_called_with()
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)
