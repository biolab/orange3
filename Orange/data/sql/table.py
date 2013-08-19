"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""

from urllib import parse
from . import postgre_backend
from .. import domain, variable, value, table, instance, filter
from Orange.data.sql import filter as sql_filter


class SqlTable(table.Table):
    base = None
    domain = None
    nrows = None
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

        self.domain = self._create_domain()
        self.name = self.table_name

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
            attr.to_sql = lambda field_name=field_name: field_name
        return attributes, metas

    #@functools.lru_cache(maxsize=128)
    def __getitem__(self, key):
        if isinstance(key, int):
            # one row
            return SqlRowInstance(
                self.domain,
                list(self.backend.query(attributes=self.domain.variables + self.domain.metas,
                                        filters=[f.to_sql()
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
        for row in self.backend.query(attributes=self.domain.variables + self.domain.metas,
                                      filters=[f.to_sql()
                                               for f in self.row_filters]):
            yield SqlRowInstance(self.domain, row)

    def copy(self):
        table = SqlTable.__new__(SqlTable)
        table.host = self.host
        table.database = self.database
        table.backend = self.backend
        table.domain = self.domain
        table.row_filters = self.row_filters
        return table

    def __len__(self):
        if self.nrows is not None:
            return self.nrows
        sql = """SELECT COUNT(*) FROM "%s" %s""" % (
            self.backend.table_name,
            self._construct_where(),
        )
        cur = self.backend.connection.cursor()
        cur.execute(sql)
        self.backend.connection.commit()
        self.nrows = cur.fetchone()[0]
        return self.nrows

    def _construct_where(self):
        filters = [f.to_sql() for f in self.row_filters]
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
        where = self._construct_where()
        return self.backend.stats(columns, where)

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
        if columns is None:
            columns = range(len(self.domain.variables))
        columns = [self.domain.variables[i].to_sql() for i in columns]

        t2 = self.copy()
        t2.row_filters += (sql_filter.IsDefinedSql(columns, negate),)
        return t2

    def _filter_has_class(self, negate=False):
        columns = [c.to_sql() for c in self.domain.class_vars]
        t2 = self.copy()
        t2.row_filters += (sql_filter.IsDefinedSql(columns, negate),)
        return t2

    def _filter_same_value(self, column, value, negate=False):
        var = self.domain[column]
        if value is None:
            pass
        elif isinstance(var, variable.DiscreteVariable):
            value = var.to_val(value)
            value = "'%s'" % var.repr_val(value)
        else:
            pass
        t2 = self.copy()
        t2.row_filters += \
            (sql_filter.SameValueSql(var.to_sql(), value, negate),)
        return t2

    def _filter_values(self, f):
        conditions = []
        for cond in f.conditions:
            var = self.domain[cond.column]
            if isinstance(cond, filter.FilterDiscrete):
                if cond.values is None:
                    values = None
                else:
                    values = ["'%s'" % var.repr_val(var.to_val(v))
                              for v in cond.values]
                new_condition = sql_filter.FilterDiscreteSql(
                    column=var.to_sql(),
                    values=values)
            elif isinstance(cond, filter.FilterContinuous):
                new_condition = sql_filter.FilterContinuousSql(
                    position=var.to_sql(),
                    oper=cond.oper,
                    ref=cond.ref,
                    max=cond.max)
            elif isinstance(cond, filter.FilterString):
                new_condition = sql_filter.FilterString(
                    var.to_sql(),
                    oper=cond.oper,
                    ref=cond.ref,
                    max=cond.max,
                    case_sensitive=cond.case_sensitive,
                )
            elif isinstance(cond, filter.FilterStringList):
                new_condition = sql_filter.FilterStringList(
                    column=var.to_sql(),
                    values=cond.values,
                    case_sensitive=cond.case_sensitive)
            else:
                raise ValueError('Invalid condition %s' % type(cond))
            conditions.append(new_condition)
        t2 = self.copy()
        t2.row_filters += (sql_filter.ValuesSql(conditions=conditions,
                                                conjunction=f.conjunction,
                                                negate=f.negate),)
        return t2


class SqlRowInstance(instance.Instance):
    def __init__(self, domain, data=None):
        super().__init__(domain, data)
        nvar = len(domain.variables)
        if len(data) > nvar:
            self._metas = data[nvar:]
