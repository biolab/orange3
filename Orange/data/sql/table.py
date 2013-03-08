"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""

from urllib import parse
import numpy as np
from . import postgre_backend
from .. import domain, storage, variable, value


class SqlTable(storage.Storage):
    base = None
    domain = None
    nrows = 0
    row_filter = None

    def __new__(cls, *args, **kwargs):
        base = kwargs.pop('base', False)
        self = super().__new__(cls)
        if base:
            self.base = base
        return self

    def __init__(self, uri, backend=None):
        if self.base:
            return

        parsed_uri = parse.urlparse(uri)

        self.host = parsed_uri.hostname
        path = parsed_uri.path.strip('/')
        self.database, self.table_name = path.split('/')

        self._init_backend(parsed_uri.scheme, backend)
        self.backend.connect(
            database=self.database,
            table=self.table_name,
            hostname=self.host,
            username=parsed_uri.username,
            password=parsed_uri.password,
        )

        self.nrows = self.backend.table_info.nrows
        self.domain = self._create_domain()

    def _init_backend(self, scheme, backend):
        if backend is not None:
            self.backend = backend
        else:
            if scheme in ('sql', 'pgsql'):
                self.backend = postgre_backend.PostgreBackend()
            else:
                raise ValueError("Unsupported schema: %s" % scheme)

    def _create_domain(self):
        attributes = self._create_attributes()
        return domain.Domain(attributes)

    def _create_attributes(self):
        attributes = []
        for name, type, values in self.backend.table_info.fields:
            if 'double' in type:
                attr = variable.ContinuousVariable(name=name)
            elif 'char' in type:
                attr = variable.DiscreteVariable(name=name, values=values)
            else:
                attr = variable.StringVariable(name=name)
            attr.get_value_from = lambda field_name=name: field_name
            attributes.append(attr)
        return attributes

    def __getitem__(self, key):
        if isinstance(key, int):
            # one row
            raise NotImplementedError
        if not isinstance(key, tuple):
            # row filter
            key = (key, Ellipsis)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")

        row_idx, col_idx = key
        if isinstance(row_idx, int):
            try:
                col_idx = self.domain.index(col_idx)
                var = self.domain[col_idx]
                return value.Value(
                    var,
                    self.backend.query(
                        self.table_name,
                        fields=var.name,
                        limit=row_idx,
                    )
                )
            except TypeError:
                pass

        # multiple rows OR single row but multiple columns:
        # construct a new table
        table = self.copy()
        table.domain = self.domain.select_columns(col_idx)
        # table.limit_rows(row_idx)
        return table

    def copy(self):
        table = SqlTable(base=self)
        table.host = self.host
        table.database = self.database
        table.backend = self.backend
        table.domain = self.domain
        table.nrows = self.nrows
        return table

    def __len__(self):
        return self.nrows
