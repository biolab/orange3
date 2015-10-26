from Orange.data import Table
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT


def check_sql_input(f):
    """
    Wrapper for widget's set_data method that first checks if the input
    is a SqlTable and:
    - if small enough, download all data and convert to Table
    - for large sql tables, show an error

    :param f: widget's `set_data` method to wrap
    :return: wrapped method that handles SQL data inputs
    """
    def new_f(self, data):
        self.error(219)
        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                self.error(219, "Download (and sample if necessary) "
                                "the SQL data first")
                data = None
        return f(self, data)

    return new_f
