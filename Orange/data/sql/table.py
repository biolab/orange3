"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""
import functools
import re
import threading
from contextlib import contextmanager
from urllib import parse

import numpy as np
import sys

import Orange.misc
psycopg2 = Orange.misc.import_late_warning("psycopg2")
psycopg2.pool = Orange.misc.import_late_warning("psycopg2.pool")

from .. import domain, variable, value, table, instance, filter,\
    DiscreteVariable, ContinuousVariable, StringVariable
from Orange.data.sql import filter as sql_filter
from Orange.data.sql.filter import CustomFilterSql
from Orange.data.sql.parser import SqlParser



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
            self, uri=None,
            host=None, database=None, user=None, password=None, schema=None,
            table=None, type_hints=None, guess_values=False, **kwargs):
        """
        Create a new proxy for sql table.

        Database connection parameters can be specified either as a string:

            table = SqlTable("user:password@host:port/database/table")

        or using a set of keyword arguments:

            table = SqlTable(database="test", table="iris")

        All but the database and table parameters are optional. Any additional
        parameters will be forwarded to the psycopg2 backend.

        If type_hints (an Orange domain) contain a column name, then
        the variable type from type_hints will be used. If it does not,
        the variable type is selected based on the column type (double
        -> ContinuousVariable, everything else -> StringVariable).
        If guess_values is True, database columns with less that 20
        different strings will become DiscreteVariables.

        Class vars and metas can be specified as a list of column names in
        __class_vars__ and __metas__ keys in type_hints dict.
        """
        assert uri is not None or database is not None

        connection_args = dict(
            host=host,
            user=user,
            password=password,
            database=database,
            schema=schema
        )
        if uri is not None:
            parameters = self.parse_uri(uri)
            table = parameters.pop("table", None)
            connection_args.update(parameters)
        connection_args.update(kwargs)

        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            1, 6, **connection_args)
        self.host = host
        self.database = database

        if table is not None:
            self.table_name = self.quote_identifier(table)
            self.domain = self.domain_from_fields(
                self._get_fields(table, guess_values=guess_values),
                type_hints=type_hints)
            self.name = table

    @classmethod
    def from_sql(
            cls, uri=None,
            host=None, database=None, user=None, password=None, schema=None,
            sql=None, type_hints=None, **kwargs):
        """
        Create a new proxy for sql select.

        Database connection parameters can be specified either as a string:

            table = SqlTable.from_sql("user:password@host:port/database/table")

        or using a set of keyword arguments:

            table = SqlTable.from_sql(database="test", sql="SELECT iris FROM iris")

        All but the database and the sql parameters are optional. Any
        additional parameters will be forwarded to the psycopg2 backend.

        If type_hints (an Orange domain) contain a column name, then
        the variable type from type_hints will be used. If it does not,
        the variable type is selected based on the column type (double
        -> ContinuousVariable, everything else -> StringVariable).

        Class vars and metas can be specified as a list of column names in
        __class_vars__ and __metas__ keys in type_hints dict.
        """
        table = cls(uri, host, database, user, password, schema, **kwargs)
        p = SqlParser(sql)
        conn = table.connection_pool.getconn()
        table.table_name = p.from_
        table.domain = table.domain_from_fields(
            p.fields_with_types(conn),
            type_hints=type_hints)
        table.connection_pool.putconn(conn)
        if p.where:
            table.row_filters = (CustomFilterSql(p.where), )

        return table

    @staticmethod
    def parse_uri(uri):
        parsed_uri = parse.urlparse(uri)
        database = parsed_uri.path.strip('/')
        if "/" in database:
            database, table = database.split('/', 1)
        else:
            table = ""

        params = parse.parse_qs(parsed_uri.query)
        for key, value in params.items():
            if len(params[key]) == 1:
                params[key] = value[0]

        params.update(dict(
            host=parsed_uri.hostname,
            port=parsed_uri.port,
            user=parsed_uri.username,
            database=database,
            password=parsed_uri.password,
        ))
        if table:
            params['table'] = table
        return params

    def domain_from_fields(self, fields, type_hints=None):
        """:fields: tuple(field_name, field_type, field_expression, values)"""
        attributes, class_vars, metas = [], [], []
        suggested_metas, suggested_class_vars = [],[]
        if type_hints != None:
            suggested_metas = [ f.name for f in type_hints.metas ]
            suggested_class_vars = [ f.name for f in type_hints.class_vars ]

        for name, field_type, field_expr, values in fields:
            var = self.var_from_field(name, field_type, field_expr, values,
                                      type_hints)

            if var.name in suggested_metas or \
                    isinstance(var, variable.StringVariable):
                metas.append(var)
            elif var.name in suggested_class_vars:
                class_vars.append(var)
            else:
                attributes.append(var)

        return domain.Domain(attributes, class_vars, metas=metas)

    @staticmethod
    def var_from_field(name, field_type, field_expr, values, type_hints):
        if type_hints != None and name in type_hints:
            var = type_hints[name]
        else:
            if any(t in field_type for t in ('double', 'numeric')):
                var = variable.ContinuousVariable(name=name)
            elif 'int' in field_type and not values:
                var = variable.ContinuousVariable(name=name)
            elif any(t in field_type for t in ('int', 'boolean')) and values:
                # TODO: make sure that int values are OK
                values = [str(val) for val in values]
                var = variable.DiscreteVariable(name=name, values=values)
                var.has_numeric_values = True
            elif (any(t in field_type for t in ('char', 'text', 'boolean'))
                  and values):
                var = variable.DiscreteVariable(name=name, values=values)
            else:
                var = variable.StringVariable(name=name)
        var.to_sql = lambda: field_expr
        return var

    def _get_fields(self, table_name, guess_values=False):
        table_name = self.unquote_identifier(table_name)
        sql = ["SELECT column_name, data_type",
               "FROM INFORMATION_SCHEMA.COLUMNS",
               "WHERE table_name =", self.quote_string(table_name),
               "ORDER BY ordinal_position"]
        with self._execute_sql_query(" ".join(sql)) as cur:
            for field, field_type in cur.fetchall():
                yield (field, field_type,
                       self.quote_identifier(field),
                       self._get_field_values(field, field_type) if guess_values else ())

    def _get_field_values(self, field_name, field_type):
        if any(t in field_type for t in ('boolean', 'int', 'char', 'text')):
            return self._get_distinct_values(field_name)
        else:
            return ()

    def _get_distinct_values(self, field_name):
        sql = " ".join(["SELECT DISTINCT", self.quote_identifier(field_name),
                        "FROM", self.table_name,
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
        table.database = self.database
        table.host = self.host
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
        if columns is not None:
            columns = [self.domain.var_from_domain(col) for col in columns]
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

    @contextmanager
    def _execute_sql_query(self, query, param=None):
        connection = self.connection_pool.getconn()
        cur = connection.cursor()
        cur.execute(query, param)
        connection.commit()
        yield cur
        self.connection_pool.putconn(connection)


class SqlRowInstance(instance.Instance):
    """
    Extends :obj:`Orange.data.Instance` to correctly handle values of meta
    attributes.
    """
    def __init__(self, domain, data=None):
        super().__init__(domain, data)
        nvar = len(domain.variables)
        if len(data) > nvar:
            self._metas = data[nvar:]
