"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""
import functools

from urllib import parse
import numpy as np
import psycopg2

from .. import domain, variable, value, table, instance, filter,\
    DiscreteVariable, ContinuousVariable
from Orange.data.sql import filter as sql_filter


class SqlTable(table.Table):
    connection = None
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
            table=None, **kwargs):
        """
        Create a new proxy for sql table.

        Database connection parameters can be specified either as a string:

            table = SqlTable("user:password@host:port/database/table")

        or using a set of keyword arguments:

            table = SqlTable(database="test", table="iris")

        All but the database and table parameters are optional. Any additional
        parameters will be forwarded to the psycopg2 backend.
        """
        assert uri is not None or table is not None

        connection_args = dict(
            host=host,
            user=user,
            password=password,
            database=database,
            schema=schema
        )
        if uri is not None:
            parameters = self.parse_uri(uri)
            table = parameters.pop("table")
            connection_args.update(parameters)
        connection_args.update(kwargs)

        self.connection = psycopg2.connect(**connection_args)
        self.host = host
        self.database = database
        self.table_name = self.name = table
        self.domain = self._create_domain()
        self.name = self.table_name

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

    def _create_domain(self):
        attributes, metas = [], []
        for name, field_type, values in self._get_fields():
            if 'double' in field_type:
                attr = variable.ContinuousVariable(name=name)
                attributes.append(attr)
            elif 'char' in field_type and values:
                attr = variable.DiscreteVariable(name=name, values=values)
                attributes.append(attr)
            else:
                attr = variable.StringVariable(name=name)
                metas.append(attr)
            field_name = '"%s"' % name
            attr.to_sql = lambda field_name=field_name: field_name

        return domain.Domain(attributes, metas=metas)

    def _get_fields(self):
        cur = self._sql_get_fields()
        for field, field_type in cur.fetchall():
            yield field, field_type, self._get_field_values(field, field_type)

    def _get_field_values(self, field_name, field_type):
        if 'double' in field_type:
            return ()
        elif 'char' in field_type:
            return self._get_distinct_values(field_name)

    def _get_distinct_values(self, field_name):
        cur = self._sql_get_distinct_values(field_name)
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
                    self.backend.query(
                        self.table_name,
                        fields=var.name,
                        limit=row_idx,
                    )
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
        filters = self.row_filters
        rows = [row_index]
        values = self._query(attributes, filters, rows)
        return SqlRowInstance(self.domain, list(values)[0])

    def __iter__(self):
        """ Iterating through the rows executes the query using a cursor and
        then yields resulting rows as SqlRowInstances as they are requested.
        """
        attributes = self.domain.variables + self.domain.metas
        filters = self.row_filters

        for row in self._query(attributes, filters):
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
        filters = [f for f in filters if f]

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
        cur = self._sql_query(fields, filters, offset=offset, limit=limit)
        while True:
            row = cur.fetchone()
            if row is None:
                break
            yield row

    def copy(self):
        """Return a copy of the SqlTable"""
        table = SqlTable.__new__(SqlTable)
        table.connection = self.connection
        table.domain = self.domain
        table.row_filters = self.row_filters
        table.table_name = table.name = self.table_name
        table.database = self.database
        table.host = self.host
        return table

    _cached__len__ = None

    def __len__(self):
        """
        Return number of rows in the table. The value is cached so it is
        computed only the first time the length is requested.
        """
        if self._cached__len__ is None:
            cur = self._count_rows()
            self._cached__len__ = cur.fetchone()[0]
        return self._cached__len__

    def _count_rows(self):
        filters = [f.to_sql() for f in self.row_filters]
        filters = [f for f in filters if f]
        return self._sql_count_rows(filters)

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
        columns = [(c.to_sql(), c.var_type == c.VarTypes.Continuous)
                   for c in columns]
        filters = [f.to_sql() for f in self.row_filters]
        filters = [f for f in filters if f]
        cur = self._sql_get_stats(columns, filters)
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
        filters = [f.to_sql() for f in self.row_filters]
        filters = [f for f in filters if f]
        dists = []
        for col in columns:
            cur = self._sql_get_distribution(col.to_sql(), filters)
            dist = np.array(cur.fetchall())
            if col.var_type == col.VarTypes.Continuous:
                dists.append((dist.T, []))
            else:
                dists.append((dist[:, 1].T, []))
        self.connection.commit()
        return dists

    def _compute_contingency(self, col_vars=None, row_var=None):
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

        filters = [f.to_sql() for f in self.row_filters]
        filters = [f for f in filters if f]

        all_contingencies = [None] * len(columns)
        for i, column in enumerate(columns):
            column_field = columns[0].to_sql()
            cur = self._sql_compute_contingency(row_field, column_field,
                                                filters)

            if column.var_type == column.VarTypes.Continuous:
                all_contingencies[i] = (self._continuous_contingencies(cur), [])
            else:
                row_mapping = {v: i for i, v in enumerate(row.values)}
                column_mapping = {v: i for i, v in enumerate(column.values)}
                all_contingencies[i] = (self._discrete_contingencies(
                    cur, row_mapping, column_mapping), [])
        return all_contingencies

    def _continuous_contingencies(self, cur):
        conts = []
        last_row_value = None
        for row_value, column_value, count in cur.fetchall():
            if row_value != last_row_value:
                if conts:
                    conts[-1] = np.array(conts[-1]).T
                conts.append(([]))
            conts[-1].append((column_value, count))
            last_row_value = row_value
        conts[-1] = np.array(conts[-1]).T
        return conts

    def _discrete_contingencies(self, cur, row_mapping, col_mapping):
        conts = np.zeros((len(row_mapping), len(col_mapping)))

        for row_value, col_value, count in cur.fetchall():
            row_index = row_mapping[row_value]
            col_index = col_mapping[col_value]
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
               "FROM", self.quote_identifier(self.table_name)]
        if filters:
            sql.extend(["WHERE", " AND ".join(filters)])
        if group_by is not None:
            sql.extend(["GROUP BY", ", ".join(group_by)])
        if order_by is not None:
            sql.extend(["ORDER BY", ",".join(order_by)])
        if offset is not None:
            sql.extend(["OFFSET", str(offset)])
        if limit is not None:
            sql.extend(["LIMIT", str(limit)])
        return self._execute_sql_query(" ".join(sql))

    def _sql_count_rows(self, filters):
        fields = ["COUNT(*)"]
        return self._sql_query(fields, filters)

    def _sql_get_fields(self):
        sql = ["SELECT column_name, data_type",
               "FROM INFORMATION_SCHEMA.COLUMNS",
               "WHERE table_name =", self.quote_string(self.table_name)]
        return self._execute_sql_query(" ".join(sql))

    def _sql_get_distinct_values(self, field_name):
        sql = ["SELECT DISTINCT", self.quote_identifier(field_name),
               "FROM", self.quote_identifier(self.table_name),
               "ORDER BY", self.quote_identifier(field_name),
               "LIMIT 21"]
        return self._execute_sql_query(" ".join(sql))

    DISCRETE_STATS = "SUM(CASE TRUE WHEN %(field_name)s IS NULL THEN 1 " \
                     "ELSE 0 END), " \
                     "SUM(CASE TRUE WHEN %(field_name)s IS NULL THEN 0 " \
                     "ELSE 1 END)"
    CONTINUOUS_STATS = "MIN(%(field_name)s), MAX(%(field_name)s), " \
                       "AVG(%(field_name)s), STDDEV(%(field_name)s), " \
                       + DISCRETE_STATS

    def _sql_get_stats(self, fields, filters):
        sql_fields = []
        for field_name, continuous in fields:
            stats = self.CONTINUOUS_STATS if continuous else self.DISCRETE_STATS
            sql_fields.append(stats % dict(field_name=field_name))
        return self._sql_query(sql_fields, filters)

    def _sql_get_distribution(self, field_name, filters):
        fields = field_name, "COUNT(%s)" % field_name
        return self._sql_query(fields, filters,
                               group_by=[field_name], order_by=[field_name])

    def _sql_compute_contingency(self, row_field, column_field, filters):
        fields = [row_field, column_field, "COUNT(%s)" % column_field]
        group_by = [row_field, column_field]
        order_by = [row_field, column_field]
        return self._sql_query(fields, filters,
                               group_by=group_by, order_by=order_by)

    def quote_identifier(self, value):
        return '"%s"' % value

    def quote_string(self, value):
        return "'%s'" % value

    def _execute_sql_query(self, sql, param=None):
        cur = self.connection.cursor()
        cur.execute(sql, param)
        self.connection.commit()
        return cur


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
