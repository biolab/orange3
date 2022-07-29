from functools import wraps

from Orange.data.dask import DaskTable, DEFAULT_SAMPLE_SIZE
from Orange.widgets.utils.messages import UnboundMsg


_sampled_dask_table = UnboundMsg(
    'The plot shows sampled data.'
)


def sample_dask_table(f):

    @wraps(f)
    def wrapper(widget, data, *args, **kwargs):
        widget.Warning.add_message('sampled_dask_table', _sampled_dask_table)
        widget.Warning.sampled_dask_table.clear()

        if isinstance(data, DaskTable):
            if len(data) > DEFAULT_SAMPLE_SIZE:
                data = data.sample()
                widget.Warning.sampled_dask_table()

            data = data.compute()
        return f(widget, data, *args, **kwargs)

    return wrapper
