"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""
import functools
import logging
import threading
import warnings
from contextlib import contextmanager
from itertools import islice
from time import strftime

import numpy as np
from Orange.data import (
    Table, Domain, Value, Instance, filter)
from Orange.data.sql import filter as sql_filter
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import TableDesc, BackendError
from Orange.misc import import_late_warning

LARGE_TABLE = 100000
AUTO_DL_LIMIT = 10000
DEFAULT_SAMPLE_TIME = 1
sql_log = logging.getLogger('sql_log')
sql_log.debug("Logging started: {}".format(strftime("%Y-%m-%d %H:%M:%S")))


class SqlTable(Table):
    table_name = None
    domain = None
    row_filters = ()

    def __new__(cls, *args, **kwargs):
        # We do not (yet) need the magic of the Table.__new__, so we call it
        # with no parameters.
        return super().__new__(cls)

    def __init__(
            self, connection_params, table_or_sql, backend=None,
            type_hints=None, inspect_values=False):
        """
        Create a new proxy for sql table.

        To create a new SqlTable, specify the connection parameters
        for psycopg2 and the name of the table/sql query used to fetch
        the data.

            table = SqlTable('database_name', 'table_name')
            table = SqlTable('database_name', 'SELECT * FROM table')

        For complex configurations, dictionary of connection parameters can
        be used instead of the database name. For documentation about
        connection parameters, see:
        http://www.postgresql.org/docs/current/static/libpq-connect.html#LIBPQ-PARAMKEYWORDS


        Data domain is inferred from the columns of the table/query.

        The (very quick) default setting is to treat all numeric columns as
        continuous variables and everything else as strings and placed among
        meta attributes.

        If inspect_values parameter is set to True, all column values are
        inspected and int/string columns with less than 21 values are
        intepreted as discrete features.

        Domains can be constructed by the caller and passed in
        type_hints parameter. Variables from the domain are used for
        the columns with the matching names; for columns without the matching
        name in the domain, types are inferred as described above.
        """
        if isinstance(connection_params, str):
            connection_params = dict(database=connection_params)

        if backend is None:
            for backend in Backend.available_backends():
                try:
                    self.backend = backend(connection_params)
                    break
                except BackendError as ex:
                    print(ex)
            else:
                raise ValueError("No backend could connect to server")
        else:
            self.backend = backend(connection_params)

        if table_or_sql is not None:
            if isinstance(table_or_sql, TableDesc):
                table = table_or_sql.sql
            elif "SELECT" in table_or_sql:
                table = "(%s) as my_table" % table_or_sql.strip("; ")
            else:
                table = self.backend.quote_identifier(table_or_sql)
            self.table_name = table
            self.domain = self.get_domain(type_hints, inspect_values)
            self.name = table

    @property
    def connection_params(self):
        warnings.warn("Use backend.connection_params", DeprecationWarning)
        return self.backend.connection_params

    def get_domain(self, type_hints=None, inspect_values=False):
        table_name = self.table_name
        if type_hints is None:
            type_hints = Domain([])

        inspect_table = table_name if inspect_values else None

        attrs, class_vars, metas = [], [], []
        for field_name, *field_metadata in self.backend.get_fields(table_name):
            var = self.backend.create_variable(field_name, field_metadata,
                                               type_hints, inspect_table)

            if var.is_string:
                metas.append(var)
            else:
                if var in type_hints.class_vars:
                    class_vars.append(var)
                elif var in type_hints.metas:
                    metas.append(var)
                else:
                    attrs.append(var)

        return Domain(attrs, class_vars, metas)

    def __getitem__(self, key):
        """ Indexing of SqlTable is performed in the following way:

        If a single row is requested, it is fetched from the database and
        returned as a SqlRowInstance.

        A new SqlTable with appropriate filters is constructed and returned
        otherwise.
        """
        if isinstance(key, int):
            # one row
            return self._fetch_row(key)

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
                return Value(
                    var,
                    next(self._query([var], rows=[row_idx]))[0]
                )
            except TypeError:
                pass

        elif not (row_idx is Ellipsis or row_idx == slice(None)):
            # TODO if row_idx specify multiple rows, one of the following must
            # happen
            #  - the new table remembers which rows are selected (implement
            #     table.limit_rows and whatever else is necessary)
            #  - return an ordinary (non-SQL) Table
            #  - raise an exception
            raise NotImplementedError("Row indices must be integers.")

        # multiple rows OR single row but multiple columns:
        # construct a new table
        table = self.copy()
        table.domain = self.domain.select_columns(col_idx)
        # table.limit_rows(row_idx)
        return table

    @functools.lru_cache(maxsize=128)
    def _fetch_row(self, row_index):
        attributes = self.domain.variables + self.domain.metas
        rows = [row_index]
        values = list(self._query(attributes, rows=rows))
        if not values:
            raise IndexError('Could not retrieve row {} from table {}'.format(
                row_index, self.name))
        return SqlRowInstance(self.domain, values[0])

    def __iter__(self):
        """ Iterating through the rows executes the query using a cursor and
        then yields resulting rows as SqlRowInstances as they are requested.
        """
        attributes = self.domain.variables + self.domain.metas

        for row in self._query(attributes):
            yield SqlRowInstance(self.domain, row)

    def _query(self, attributes=None, filters=(), rows=None):
        if attributes is not None:
            fields = []
            for attr in attributes:
                assert hasattr(attr, 'to_sql'), \
                    "Cannot use ordinary attributes with sql backend"
                field_str = '(%s) AS "%s"' % (attr.to_sql(), attr.name)
                fields.append(field_str)
            if not fields:
                raise ValueError("No fields selected.")
        else:
            fields = ["*"]

        filters = [f.to_sql() for f in filters]

        offset = limit = None
        if rows is not None:
            if isinstance(rows, slice):
                offset = rows.start or 0
                if rows.stop is not None:
                    limit = rows.stop - offset
            else:
                rows = list(rows)
                offset, stop = min(rows), max(rows)
                limit = stop - offset + 1

        # TODO: this returns all rows between min(rows) and max(rows): fix!
        query = self._sql_query(fields, filters, offset=offset, limit=limit)
        with self.backend.execute_sql_query(query) as cur:
            while True:
                row = cur.fetchone()
                if row is None:
                    break
                yield row

    def copy(self):
        """Return a copy of the SqlTable"""
        table = SqlTable.__new__(SqlTable)
        table.backend = self.backend
        table.domain = self.domain
        table.row_filters = self.row_filters
        table.table_name = self.table_name
        table.name = self.name
        return table

    def __bool__(self):
        """Return True if the SqlTable is not empty."""
        query = self._sql_query(["1"], limit=1)
        with self.backend.execute_sql_query(query) as cur:
            return cur.fetchone() is not None

    _cached__len__ = None

    def __len__(self):
        """
        Return number of rows in the table. The value is cached so it is
        computed only the first time the length is requested.
        """
        if self._cached__len__ is None:
            return self._count_rows()
        return self._cached__len__

    def _count_rows(self):
        query = self._sql_query(["COUNT(*)"])
        with self.backend.execute_sql_query(query) as cur:
            self._cached__len__ = cur.fetchone()[0]
        return self._cached__len__

    def approx_len(self, get_exact=False):
        if self._cached__len__ is not None:
            return self._cached__len__

        approx_len = None
        try:
            query = self._sql_query(["*"])
            approx_len = self.backend.count_approx(query)
            if get_exact:
                threading.Thread(target=len, args=(self,)).start()
        except NotImplementedError:
            pass

        if approx_len is None:
            approx_len = len(self)

        return approx_len

    _X = None
    _Y = None
    _metas = None
    _W = None
    _ids = None

    def download_data(self, limit=None, partial=False):
        """Download SQL data and store it in memory as numpy matrices."""
        if limit and not partial and self.approx_len() > limit:
            raise ValueError("Too many rows to download the data into memory.")
        X = [np.empty((0, len(self.domain.attributes)))]
        Y = [np.empty((0, len(self.domain.class_vars)))]
        metas = [np.empty((0, len(self.domain.metas)))]
        for row in islice(self, limit):
            X.append(row._x)
            Y.append(row._y)
            metas.append(row._metas)
        self._X = np.vstack(X).astype(np.float64)
        self._Y = np.vstack(Y).astype(np.float64)
        self._metas = np.vstack(metas).astype(object)
        self._W = np.empty((self._X.shape[0], 0))
        self._init_ids(self)
        if not partial or limit and self._X.shape[0] < limit:
            self._cached__len__ = self._X.shape[0]

    @property
    def X(self):
        """Numpy array with attribute values."""
        if self._X is None:
            self.download_data(AUTO_DL_LIMIT)
        return self._X

    @property
    def Y(self):
        """Numpy array with class values."""
        if self._Y is None:
            self.download_data(AUTO_DL_LIMIT)
        return self._Y

    @property
    def metas(self):
        """Numpy array with class values."""
        if self._metas is None:
            self.download_data(AUTO_DL_LIMIT)
        return self._metas

    @property
    def W(self):
        """Numpy array with class values."""
        if self._W is None:
            self.download_data(AUTO_DL_LIMIT)
        return self._W

    @property
    def ids(self):
        """Numpy array with class values."""
        if self._ids is None:
            self.download_data(AUTO_DL_LIMIT)
        return self._ids

    @ids.setter
    def ids(self, value):
        self._ids = value

    @ids.deleter
    def ids(self):
        del self._ids

    def has_weights(self):
        return False

    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_var=False):
        if self.approx_len() > LARGE_TABLE:
            self = self.sample_time(DEFAULT_SAMPLE_TIME)

        if columns is not None:
            columns = [self.domain[col] for col in columns]
        else:
            columns = list(self.domain)
            if include_metas:
                columns += list(self.domain.metas)
        return self._get_stats(columns)

    def _get_stats(self, columns):
        columns = [(c.to_sql(), c.is_continuous) for c in columns]
        sql_fields = []
        for field_name, continuous in columns:
            stats = self.CONTINUOUS_STATS if continuous else self.DISCRETE_STATS
            sql_fields.append(stats % dict(field_name=field_name))
        query = self._sql_query(sql_fields)
        with self.backend.execute_sql_query(query) as cur:
            results = cur.fetchone()
        stats = []
        i = 0
        for ci, (field_name, continuous) in enumerate(columns):
            if continuous:
                stats.append(results[i:i+6])
                i += 6
            else:
                stats.append((None,) * 4 + results[i:i+2])
                i += 2
        return stats

    def _compute_distributions(self, columns=None):
        if self.approx_len() > LARGE_TABLE:
            self = self.sample_time(DEFAULT_SAMPLE_TIME)

        if columns is not None:
            columns = [self.domain[col] for col in columns]
        else:
            columns = list(self.domain)
        return self._get_distributions(columns)

    def _get_distributions(self, columns):
        dists = []
        for col in columns:
            field_name = col.to_sql()
            fields = field_name, "COUNT(%s)" % field_name
            query = self._sql_query(fields,
                                    filters=['%s IS NOT NULL' % field_name],
                                    group_by=[field_name],
                                    order_by=[field_name])
            with self.backend.execute_sql_query(query) as cur:
                dist = np.array(cur.fetchall())
            if col.is_continuous:
                dists.append((dist.T, []))
            else:
                dists.append((dist[:, 1].T, []))
        return dists

    def _compute_contingency(self, col_vars=None, row_var=None):
        if self.approx_len() > LARGE_TABLE:
            self = self.sample_time(DEFAULT_SAMPLE_TIME)

        if col_vars is None:
            col_vars = range(len(self.domain.variables))
        if len(col_vars) != 1:
            raise NotImplementedError("Contingency for multiple columns "
                                      "has not yet been implemented.")
        if row_var is None:
            raise NotImplementedError("Defaults have not been implemented yet")

        row = self.domain[row_var]
        if not row.is_discrete:
            raise TypeError("Row variable must be discrete")

        columns = [self.domain[var] for var in col_vars]

        if any(not (var.is_continuous or var.is_discrete)
               for var in columns):
            raise ValueError("contingency can be computed only for discrete "
                             "and continuous values")

        row_field = row.to_sql()

        all_contingencies = [None] * len(columns)
        for i, column in enumerate(columns):
            column_field = column.to_sql()
            fields = [row_field, column_field, "COUNT(%s)" % column_field]
            group_by = [row_field, column_field]
            order_by = [column_field]
            filters = ['%s IS NOT NULL' % f
                       for f in (row_field, column_field)]
            query = self._sql_query(fields, filters=filters,
                                    group_by=group_by, order_by=order_by)
            with self.backend.execute_sql_query(query) as cur:
                data = list(cur.fetchall())
                if column.is_continuous:
                    all_contingencies[i] = \
                        (self._continuous_contingencies(data, row), [])
                else:
                    all_contingencies[i] =\
                        (self._discrete_contingencies(data, row, column), [])
        return all_contingencies, None

    def _continuous_contingencies(self, data, row):
        values = np.zeros(len(data))
        counts = np.zeros((len(row.values), len(data)))
        last = None
        i = -1
        for row_value, column_value, count in data:
            if column_value == last:
                counts[row.to_val(row_value), i] += count
            else:
                i += 1
                last = column_value
                values[i] = column_value
                counts[row.to_val(row_value), i] += count
        return (values, counts)

    def _discrete_contingencies(self, data, row, column):
        conts = np.zeros((len(row.values), len(column.values)))
        for row_value, col_value, count in data:
            row_index = row.to_val(row_value)
            col_index = column.to_val(col_value)
            conts[row_index, col_index] = count
        return conts

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
        columns = [self.domain[i].to_sql() for i in columns]

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
        elif var.is_discrete:
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

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        assert row_indices is ...

        table = source.copy()
        table.domain = domain
        return table

    # sql queries
    def _sql_query(self, fields, filters=(),
                   group_by=None, order_by=None, offset=None, limit=None,
                   use_time_sample=None):

        row_filters = [f.to_sql() for f in self.row_filters]
        row_filters.extend(filters)
        return self.backend.create_sql_query(
            self.table_name, fields, row_filters, group_by, order_by,
            offset, limit, use_time_sample)


    DISCRETE_STATS = "SUM(CASE TRUE WHEN %(field_name)s IS NULL THEN 1 " \
                     "ELSE 0 END), " \
                     "SUM(CASE TRUE WHEN %(field_name)s IS NULL THEN 0 " \
                     "ELSE 1 END)"
    CONTINUOUS_STATS = "MIN(%(field_name)s)::double precision, " \
                       "MAX(%(field_name)s)::double precision, " \
                       "AVG(%(field_name)s)::double precision, " \
                       "STDDEV(%(field_name)s)::double precision, " \
                       + DISCRETE_STATS

    def sample_percentage(self, percentage, no_cache=False):
        if percentage >= 100:
            return self
        return self._sample('system', percentage,
                            no_cache=no_cache)

    def sample_time(self, time_in_seconds, no_cache=False):
        return self._sample('system_time', int(time_in_seconds * 1000),
                            no_cache=no_cache)

    def _sample(self, method, parameter, no_cache=False):
        import psycopg2
        if "," in self.table_name:
            raise NotImplementedError("Sampling of complex queries is not supported")

        parameter = str(parameter)
        if "." in self.table_name:
            schema, name = self.table_name.split(".")
            sample_name = '__%s_%s_%s' % (
                self.backend.unquote_identifier(name),
                method,
                parameter.replace('.', '_').replace('-', '_'))
            sample_table_q = ".".join([schema, self.backend.quote_identifier(sample_name)])
        else:
            sample_table = '__%s_%s_%s' % (
                self.backend.unquote_identifier(self.table_name),
                method,
                parameter.replace('.', '_').replace('-', '_'))
            sample_table_q = self.backend.quote_identifier(sample_table)
        create = False
        try:
            query = "SELECT * FROM " + sample_table_q + " LIMIT 0;"
            with self.backend.execute_sql_query(query): pass

            if no_cache:
                query = "DROP TABLE " + sample_table_q
                with self.backend.execute_sql_query(query): pass
                create = True

        except BackendError:
            create = True

        if create:
            with self.backend.execute_sql_query(" ".join([
                    "CREATE TABLE", sample_table_q, "AS",
                    "SELECT * FROM", self.table_name,
                    "TABLESAMPLE", method, "(", parameter, ")"])):
                pass

        sampled_table = self.copy()
        sampled_table.table_name = sample_table_q
        with sampled_table.backend.execute_sql_query(
                'ANALYZE' + sample_table_q):
            pass
        return sampled_table

    @contextmanager
    def _execute_sql_query(self, query, param=None):
        warnings.warn("Use backend.execute_sql_query", DeprecationWarning)
        with self.backend.execute_sql_query(query, param) as cur:
            yield cur

    def checksum(self, include_metas=True):
        return np.nan


class SqlRowInstance(Instance):
    """
    Extends :obj:`Orange.data.Instance` to correctly handle values of meta
    attributes.
    """

    def __init__(self, domain, data=None):
        nvar = len(domain.variables)
        super().__init__(domain, data[:nvar])
        if len(data) > nvar:
            self._metas = np.asarray(data[nvar:], dtype=object)
