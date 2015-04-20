"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""
import functools
import re
import threading
from contextlib import contextmanager

import numpy as np

import Orange.misc
psycopg2 = Orange.misc.import_late_warning("psycopg2")
psycopg2.pool = Orange.misc.import_late_warning("psycopg2.pool")

from .. import domain, variable, value, table, instance, filter,\
    DiscreteVariable, ContinuousVariable, StringVariable
from Orange.data.sql import filter as sql_filter

LARGE_TABLE = 100000
DEFAULT_SAMPLE_TIME = 1


class SqlTable(table.Table):
    connection_pool = None
    table_name = None
    domain = None
    row_filters = ()

    def __new__(cls, *args, **kwargs):
        # We do not (yet) need the magic of the Table.__new__, so we call it
        # with no parameters.
        return super().__new__(cls)

    def __init__(
            self, connection_params, table_or_sql,
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
        self.connection_params = connection_params

        if self.connection_pool is None:
            self.create_connection_pool()

        if table_or_sql is not None:
            if "SELECT" in table_or_sql.upper():
                table = "(%s) as my_table" % table_or_sql.strip("; ")
            else:
                table = self.quote_identifier(table_or_sql)
            self.table_name = table
            self.domain = self.get_domain(type_hints, inspect_values)
            self.name = table

    def create_connection_pool(self):
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            1, 16, **self.connection_params)

    def get_domain(self, type_hints=None, guess_values=False):
        if type_hints is None:
            type_hints = domain.Domain([])

        fields = []
        query = "SELECT * FROM %s LIMIT 0" % self.table_name
        with self._execute_sql_query(query) as cur:
            for col in cur.description:
                fields.append(col)

        def add_to_sql(var, field_name):
            if isinstance(var, ContinuousVariable):
                var.to_sql = ToSql("({})::double precision".format(
                    self.quote_identifier(field_name)))
            elif isinstance(var, DiscreteVariable):
                var.to_sql = ToSql("({})::text".format(
                    self.quote_identifier(field_name)))
            else:
                var.to_sql = ToSql(self.quote_identifier(field_name))

        attrs, class_vars, metas = [], [], []
        for field_name, type_code, *rest in fields:
            if field_name in type_hints:
                var = type_hints[field_name]
            else:
                var = self.get_variable(field_name, type_code, guess_values)
            add_to_sql(var, field_name)

            if isinstance(var, StringVariable):
                metas.append(var)
            else:
                if var in type_hints.class_vars:
                    class_vars.append(var)
                elif var in type_hints.metas:
                    metas.append(var)
                else:
                    attrs.append(var)

        return domain.Domain(attrs, class_vars, metas)

    def get_variable(self, field_name, type_code, inspect_values=False):
        FLOATISH_TYPES = (700, 701, 1700)  # real, float8, numeric
        INT_TYPES = (20, 21, 23)  # bigint, int, smallint
        CHAR_TYPES = (25, 1042, 1043,)  # text, char, varchar
        BOOLEAN_TYPES = (16,)  # bool

        if type_code in FLOATISH_TYPES:
            return ContinuousVariable(field_name)

        if type_code in INT_TYPES:  # bigint, int, smallint
            if inspect_values:
                values = self.get_distinct_values(field_name)
                if values:
                    return DiscreteVariable(field_name, values)
            return ContinuousVariable(field_name)

        if type_code in BOOLEAN_TYPES:
            return DiscreteVariable(field_name, ['false', 'true'])

        if type_code in CHAR_TYPES:
            if inspect_values:
                values = self.get_distinct_values(field_name)
                if values:
                    return DiscreteVariable(field_name, values)

        return StringVariable(field_name)

    def get_distinct_values(self, field_name):
        sql = " ".join(["SELECT DISTINCT (%s)::text" %
                            self.quote_identifier(field_name),
                        "FROM", self.table_name,
                        "WHERE {} IS NOT NULL".format(
                            self.quote_identifier(field_name)),
                        "ORDER BY", self.quote_identifier(field_name),
                        "LIMIT 21"])
        with self._execute_sql_query(sql) as cur:
            values = cur.fetchall()
        if len(values) > 20:
            return ()
        else:
            return tuple(x[0] for x in values)

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
                return value.Value(
                    var,
                    self._query(self.table_name, var, rows=[row_idx])
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
        values = list(self._query(attributes, rows=rows))[0]
        return SqlRowInstance(self.domain, values)

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
        with self._execute_sql_query(query) as cur:
            while True:
                row = cur.fetchone()
                if row is None:
                    break
                yield row

    def copy(self):
        """Return a copy of the SqlTable"""
        table = SqlTable.__new__(SqlTable)
        table.connection_pool = self.connection_pool
        table.domain = self.domain
        table.row_filters = self.row_filters
        table.table_name = self.table_name
        table.name = self.name
        table.connection_params = self.connection_params
        return table

    def __bool__(self):
        """Return True if the SqlTable is not empty."""
        query = self._sql_query(["1"], limit=1)
        with self._execute_sql_query(query) as cur:
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
        with self._execute_sql_query(query) as cur:
            self._cached__len__ = cur.fetchone()[0]
        return self._cached__len__

    def approx_len(self, get_exact=False):
        if self._cached__len__ is not None:
            return self._cached__len__
        sql = "EXPLAIN " + self._sql_query(["*"])
        with self._execute_sql_query(sql) as cur:
            s = ''.join(row[0] for row in cur.fetchall())
        alen = int(re.findall('rows=(\d*)', s)[0])
        if get_exact:
            threading.Thread(target=len, args=(self,)).start()
        return alen

    _X = None
    _Y = None

    def download_data(self, limit=None):
        """Download SQL data and store it in memory as numpy matrices."""
        if limit and len(self) > limit: #TODO: faster check for size limit
            raise ValueError("Too many rows to download the data into memory.")
        self._X = np.vstack(row._x for row in self)
        self._Y = np.vstack(row._y for row in self)
        self._cached__len__ = self._X.shape[0]

    @property
    def X(self):
        """Numpy array with attribute values."""
        if self._X is None:
            self.download_data(1000)
        return self._X

    @property
    def Y(self):
        """Numpy array with class values."""
        if self._Y is None:
            self.download_data(1000)
        return self._Y

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
        columns = [(c.to_sql(), isinstance(c, ContinuousVariable))
                   for c in columns]
        sql_fields = []
        for field_name, continuous in columns:
            stats = self.CONTINUOUS_STATS if continuous else self.DISCRETE_STATS
            sql_fields.append(stats % dict(field_name=field_name))
        query = self._sql_query(sql_fields)
        with self._execute_sql_query(query) as cur:
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
            with self._execute_sql_query(query) as cur:
                dist = np.array(cur.fetchall())
            if isinstance(col, ContinuousVariable):
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
        if not isinstance(row, DiscreteVariable):
            raise TypeError("Row variable must be discrete")

        columns = [self.domain[var] for var in col_vars]

        if any(not isinstance(var, (ContinuousVariable, DiscreteVariable))
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
            with self._execute_sql_query(query) as cur:
                data = list(cur.fetchall())
                if isinstance(column, ContinuousVariable):
                    all_contingencies[i] = \
                        (self._continuous_contingencies(data, row), [])
                else:
                    all_contingencies[i] =\
                        (self._discrete_contingencies(data, row, column), [])
        return all_contingencies

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

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        assert row_indices is ...

        table = source.copy()
        table.domain = domain
        return table

    # sql queries
    def _sql_query(self, fields, filters=(),
                   group_by=None, order_by=None, offset=None, limit=None):
        sql = ["SELECT", ', '.join(fields),
               "FROM", self.table_name]
        row_filters = [f.to_sql() for f in self.row_filters]
        row_filters.extend(filters)
        if row_filters:
            sql.extend(["WHERE", " AND ".join(row_filters)])
        if group_by is not None:
            sql.extend(["GROUP BY", ", ".join(group_by)])
        if order_by is not None:
            sql.extend(["ORDER BY", ",".join(order_by)])
        if offset is not None:
            sql.extend(["OFFSET", str(offset)])
        if limit is not None:
            sql.extend(["LIMIT", str(limit)])
        return " ".join(sql)

    DISCRETE_STATS = "SUM(CASE TRUE WHEN %(field_name)s IS NULL THEN 1 " \
                     "ELSE 0 END), " \
                     "SUM(CASE TRUE WHEN %(field_name)s IS NULL THEN 0 " \
                     "ELSE 1 END)"
    CONTINUOUS_STATS = "MIN(%(field_name)s)::double precision, " \
                       "MAX(%(field_name)s)::double precision, " \
                       "AVG(%(field_name)s)::double precision, " \
                       "STDDEV(%(field_name)s)::double precision, " \
                       + DISCRETE_STATS

    def quote_identifier(self, value):
        return '"%s"' % value

    def unquote_identifier(self, value):
        if value.startswith('"'):
            return value[1:len(value)-1]
        else:
            return value

    def quote_string(self, value):
        return "'%s'" % value

    def sample_percentage(self, percentage, no_cache=False):
        return self._sample('blocksample_percent', percentage,
                            no_cache=no_cache)

    def sample_time(self, time_in_seconds, no_cache=False):
        return self._sample('blocksample_time', int(time_in_seconds * 1000),
                            no_cache=no_cache)

    def _sample(self, method, parameter, no_cache=False):
        if "," in self.table_name:
            raise NotImplementedError("Sampling of complex queries is not supported")

        sample_table = '__%s_%s_%s' % (
            self.unquote_identifier(self.table_name),
            method,
            str(parameter).replace('.', '_'))
        create = False
        try:
            with self._execute_sql_query("SELECT * FROM %s LIMIT 0" % self.quote_identifier(sample_table)) as cur:
                cur.fetchall()

            if no_cache:
                with self._execute_sql_query("DROP TABLE %s" % self.quote_identifier(sample_table)) as cur:
                    cur.fetchall()
                create = True

        except psycopg2.ProgrammingError:
            create = True

        if create:
            with self._execute_sql_query('SELECT %s(%s, %s, %s)' % (
                    method,
                    self.quote_string(sample_table),
                    self.quote_string(self.unquote_identifier(self.table_name)),
                    parameter)) as cur:
                cur.fetchall()

        sampled_table = self.copy()
        sampled_table.table_name = self.quote_identifier(sample_table)
        return sampled_table

    @contextmanager
    def _execute_sql_query(self, query, param=None):
        connection = self.connection_pool.getconn()
        cur = connection.cursor()
        try:
            cur.execute(query, param)
            yield cur
        finally:
            connection.commit()
            self.connection_pool.putconn(connection)

    def checksum(self, include_metas=True):
        return np.nan

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('connection_pool')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.create_connection_pool()


class SqlRowInstance(instance.Instance):
    """
    Extends :obj:`Orange.data.Instance` to correctly handle values of meta
    attributes.
    """

    def __init__(self, domain, data=None):
        nvar = len(domain.variables)
        super().__init__(domain, data[:nvar])
        if len(data) > nvar:
            self._metas = data[nvar:]


class ToSql:
    def __init__(self, sql):
        self.sql = sql

    def __call__(self):
        return self.sql
