"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""

from urllib import parse
import functools
import numpy as np
from . import postgre_backend
from .. import domain, storage, variable, value, table, instance
from Orange.data.sql.filter import IsDefinedSql


class SqlTable(table.Table):
    base = None
    domain = None
    nrows = 0
    row_filters = ()

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, uri=None,
                 host=None, database=None, user=None, password=None, table=None,
                 backend=None):
        if self.base:
            return

        assert uri is not None or table is not None

        if uri is not None:
            parsed_uri = parse.urlparse(uri)
            host = parsed_uri.hostname
            path = parsed_uri.path.strip('/')
            database, table = path.split('/')
            user = parsed_uri.username
            password = parsed_uri.password
        scheme = 'pgsql'
        self.host = host
        self.database = database
        self.table_name = table

        self._init_backend(scheme, backend)
        self.backend.connect(
            database=self.database,
            table=self.table_name,
            hostname=self.host,
            username=user,
            password=password,
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
        attributes, metas = self._create_attributes()
        return domain.Domain(attributes, metas=metas)

    def _create_attributes(self):
        attributes, metas = [], []
        for name, type, values in self.backend.table_info.fields:
            if 'double' in type:
                attr = variable.ContinuousVariable(name=name)
                attributes.append(attr)
            elif 'char' in type and values:
                attr = variable.DiscreteVariable(name=name, values=values)
                attributes.append(attr)
            else:
                attr = variable.StringVariable(name=name)
                metas.append(attr)
            field_name = '"%s"' % name
            attr.get_value_from = lambda field_name=field_name: field_name
        return attributes, metas

    #@functools.lru_cache(maxsize=128)
    def __getitem__(self, key):
        if isinstance(key, int):
            # one row
            return self._create_row_instance(
                list(self.backend.query(filters=[f.to_sql(self)
                                                 for f in self.row_filters],
                                        rows=[key]))[0])
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

    def __iter__(self):
        for row in self.backend.query(filters=[f.to_sql(self)
                                               for f in self.row_filters]):
            yield self._create_row_instance(row)

    def _create_row_instance(self, row):
        if getattr(self, 'discrete_variables', None) is None:
            self.discrete_variables = {
                var: dict(zip(var.values, range(len(var.values))))
                for var in self.domain.variables
                if isinstance(var, variable.DiscreteVariable)
            }
        row = list(row)
        for (idx, value), var in zip(enumerate(row), self.domain.variables):
            if value is None:
                row[idx] = float('nan')
            elif var in self.discrete_variables:
                row[idx] = self.discrete_variables[var][value]
        return SqlRowInstance(self, row)

    def copy(self):
        table = SqlTable.__new__(SqlTable)
        table.host = self.host
        table.database = self.database
        table.backend = self.backend
        table.domain = self.domain
        table.nrows = None
        table.row_filters = self.row_filters
        return table

    def __len__(self):
        if self.nrows is not None:
            return self.nrows
        cur = self.backend.connection.cursor()
        cur.execute("""SELECT COUNT(*) FROM "%s" %s""" % (
            self.backend.table_name,
            self._construct_where(),
        ))
        self.backend.connection.commit()
        self.nrows = cur.fetchone()[0]
        return self.nrows

    def _construct_where(self):
        filters = [f.to_sql(self) for f in self.row_filters]
        filters = [f for f in filters if f]
        if filters:
            return " WHERE %s " % " AND ".join(filters)
        else:
            return ""

    def has_weights(self):
        return False

    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_var=False):
        if columns is not None:
            columns = [self.domain.var_from_domain(col) for col in columns]
        else:
            columns = list(self.domain)
            if include_metas:
                columns += list(self.domain.metas)
        return self.backend.stats(columns)

    def _compute_distributions(self, columns=None):
        if columns is not None:
            columns = [self.domain.var_from_domain(col) for col in columns]
        else:
            columns = list(self.domain)
        return self.backend.distributions(columns)

    def X_density(self):
        return self.DENSE

    def Y_density(self):
        return self.DENSE

    def metas_density(self):
        return self.DENSE

    # Filters
    def _filter_is_defined(self, columns=None, negate=False):
        t2 = self.copy()
        t2.row_filters += (IsDefinedSql(columns, negate),)
        return t2


class SqlRowInstance(instance.Instance):
    def __init__(self, table, row):
        """
        Construct a data instance representing the given row of the table.
        """
        super().__init__(table.domain)

        self._x = self._values = row

        self.table = table
