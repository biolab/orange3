"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""

from urllib import parse
import numpy as np
from . import postgre_backend
from .. import domain, storage, variable, value, table, instance


class SqlTable(table.Table):
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
            return SqlRowInstance(self, key)
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

    def has_weights(self):
        return False

    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_var=False):
        if columns is not None:
            columns = [self.domain.var_from_domain(col) for col in columns]
        else:
            columns = list(self.domain)
        return self.backend.stats(columns[:4])

    def X_density(self):
        return self.DENSE

    def Y_density(self):
        return self.DENSE

    def metas_density(self):
        return self.DENSE


class SqlRowInstance(instance.Instance):
    def __init__(self, table, row_index):
        """
        Construct a data instance representing the given row of the table.
        """
        super().__init__(table.domain)
        self._x = self._values = table.backend.query(rows=[row_index])[0]

        self.row_index = row_index
        self.table = table


    @property
    def weight(self):
        if not self.table.has_weights():
            return 1
        return self.table._W[self.row_index]

    @staticmethod
    def sp_values(matrix, row, variables):
        if sp.issparse(matrix):
            begptr, endptr = matrix.indptr[row:row + 2]
            rendptr = min(endptr, begptr + 5)
            variables = [variables[var]
                         for var in matrix.indices[begptr:rendptr]]
            s = ", ".join("{}={}".format(var.name, var.str_val(val))
                for var, val in zip(variables, matrix.data[begptr:rendptr]))
            if rendptr != endptr:
                s += ", ..."
            return s
        else:
            return Instance.str_values(matrix[row], variables)

    def __str__(self):
        table = self.table
        domain = table.domain
        row = self.row_index
        s = "[" + self.sp_values(table._X, row, domain.attributes)
        if domain.class_vars:
            s += " | " + self.sp_values(table._Y, row, domain.class_vars)
        s += "]"
        if self._domain.metas:
            s += " {" + self.sp_values(table._metas, row, domain.metas) + "}"
        return s

    __repr__ = __str__
