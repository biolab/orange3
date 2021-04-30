from functools import wraps

from Orange.widgets.utils.messages import UnboundMsg
from Orange.data import Table
from Orange.data.sql.table import SqlTable, AUTO_DL_LIMIT

_download_sql_data = UnboundMsg(
    "Download (and sample if necessary) the SQL data first")


def check_sql_input(f):
    """
    Wrapper for widget's set_data method that first checks if the input
    is a SqlTable and:
    - if small enough, download all data and convert to Table
    - for large sql tables, show an error

    :param f: widget's `set_data` method to wrap
    :return: wrapped method that handles SQL data inputs
    """
    @wraps(f)
    def new_f(widget, data, *args, **kwargs):
        widget.Error.add_message("download_sql_data", _download_sql_data)
        widget.Error.download_sql_data.clear()
        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                widget.Error.download_sql_data()
                data = None
        return f(widget, data, *args, **kwargs)

    return new_f


def check_sql_input_sequence(f):
    """
    Wrapper for widget's set_data/insert_data methodss that first checks
    if the input is a SqlTable and:
    - if small enough, download all data and convert to Table
    - for large sql tables, show an error

    :param f: widget's `set_data` method to wrap
    :return: wrapped method that handles SQL data inputs
    """
    @wraps(f)
    def new_f(widget, index, data, *args, **kwargs):
        widget.Error.add_message("download_sql_data", _download_sql_data)
        widget.Error.download_sql_data.clear()
        if isinstance(data, SqlTable):
            if data.approx_len() < AUTO_DL_LIMIT:
                data = Table(data)
            else:
                widget.Error.download_sql_data()
                data = None
        return f(widget, index, data, *args, **kwargs)

    return new_f
