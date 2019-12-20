import unittest

from Orange.data.sql.backend.base import BackendError
try:
    from Orange.data.sql.backend.mssql import PymssqlBackend, parse_ex
    err = ""
except ModuleNotFoundError as ex:
    PymssqlBackend = None
    err = str(ex)


class TestPymssqlBackend(unittest.TestCase):
    @unittest.skipIf(PymssqlBackend is None, err)
    def test_connection_error(self):
        connection_params = {"host": "host", "port": "", "database": "DB"}
        self.assertRaises(BackendError, PymssqlBackend, connection_params)

    @unittest.skipIf(PymssqlBackend is None, err)
    def test_parse_ex(self):
        err_msg = "Foo"
        self.assertEqual(parse_ex(ValueError(err_msg)), err_msg)


if __name__ == '__main__':
    unittest.main()
