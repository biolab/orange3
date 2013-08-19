"""
Support for example tables wrapping data stored on a PostgreSQL server.
"""

from urllib import parse
import numpy as np
import psycopg2

from .. import domain, variable, value, table, instance, filter
from Orange.data.sql import filter as sql_filter


class SqlTable(table.Table):
    connection = None
    table_name = None
    domain = None
    row_filters = ()

    def __new__(cls, *args, **kwargs):
        """We do not (yet) need the magic of the Table.__new__, so we call it
        with no parameters.
        """
        return super().__new__(cls)

    def __init__(
            self, uri=None,
            host=None, database=None, user=None, password=None, schema=None,
            table=None, **kwargs):
        """
        Create a new  proxy for sql table.

        Database connection parameters can be specified either as a string:

            table = SqlTable("user:password@host:port/database/table?schema=dbschema")

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
            parameters = self._parse_uri(uri)
            table = parameters.pop("table")
            connection_args.update(parameters)

        self.connection = psycopg2.connect(**connection_args)
        self.table_name = self.name = table
        self.domain = self._create_domain()
        self.name = self.table_name

    def _parse_uri(self, uri):
        parsed_uri = parse.urlparse(uri)
        path = parsed_uri.path.strip('/')
        database, table = path.split('/')

        return dict(
            host=parsed_uri.hostname,
            port=parsed_uri.port,
            user=parsed_uri.username,
            database=database,
            password=parsed_uri.password,
            table=table,
        )
        # TODO: parse schema

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
        cur = self.connection.cursor()
        cur.execute("""
            SELECT column_name, data_type
              FROM INFORMATION_SCHEMA.COLUMNS
             WHERE table_name = %s;""", (self.table_name,))
        self.connection.commit()

        for field, field_type in cur.fetchall():
            yield field, field_type, self._get_field_values(field, field_type)

    def _get_field_values(self, field_name, field_type):
        if 'double' in field_type:
            return ()
        elif 'char' in field_type:
            return self._get_distinct_values(field_name)

    def _get_distinct_values(self, field_name):
        cur = self.connection.cursor()
        cur.execute("""SELECT DISTINCT "%s" FROM "%s" ORDER BY %s LIMIT 21""" %
                    (field_name, self.table_name, field_name))
        self.connection.commit()
        values = cur.fetchall()

        if len(values) > 20:
            return ()
        else:
            return tuple(x[0] for x in values)

    #@functools.lru_cache(maxsize=128)
    def __getitem__(self, key):
        """ Indexing of SqlTable is performed in the following way:

        If a single row is requested, it is fetched from the database and
        returned as a SqlRowInstance.

        A new SqlTable with appropriate filters is constructed and returned
        otherwise.
        """
        if isinstance(key, int):
            # one row
            return SqlRowInstance(
                self.domain,
                list(self._query(attributes=self.domain.variables + self.domain.metas,
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
        """ Iterating through the rows executes the query using a cursor and
        then yields resulting rows as SqlRowInstances as they are requested.
        """
        for row in self._query(attributes=self.domain.variables + self.domain.metas,
                               filters=[f.to_sql()
                                        for f in self.row_filters]):
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

        sql = """SELECT %s FROM "%s" """ % (', '.join(fields), self.table_name)
        filters = [f for f in filters if f]
        if filters:
            sql += " WHERE %s " % " AND ".join(filters)
        if rows is not None:
            if isinstance(rows, slice):
                start = rows.start or 0
                if rows.stop is None:
                    sql += " OFFSET %d" % start
                else:
                    sql += " OFFSET %d LIMIT %d" % (start, rows.stop - start)
            else:
                rows = list(rows)
                start, stop = min(rows), max(rows)
                sql += " OFFSET %d LIMIT %d" % (start, stop - start + 1)
        cur = self.connection.cursor()
        cur.execute(sql)
        self.connection.commit()
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
        table.table_name = self.table_name
        return table

    _cached__len__ = None

    def __len__(self):
        """
        Return number of rows in the table. The value is cached so it is
        computed only the first time the length is requested.
        """
        if self._cached__len__ is None:
            sql = """SELECT COUNT(*) FROM "%s" %s""" % (
                self.table_name,
                self._construct_where(),
            )
            cur = self.connection.cursor()
            cur.execute(sql)
            self.connection.commit()
            self._cached__len__ = cur.fetchone()[0]
        return self._cached__len__

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
        return self._get_stats(columns, where)

    def _get_stats(self, columns, where=""):
        stats = []
        for column in columns:
            if column.var_type == column.VarTypes.Continuous:
                column = column.to_sql()
                stats.append(", ".join((
                    "MIN(%s)" % column,
                    "MAX(%s)" % column,
                    "AVG(%s)" % column,
                    "STDDEV(%s)" % column,
                    #"0",
                    "SUM(CASE TRUE"
                    "       WHEN %s IS NULL THEN 1"
                    "       ELSE 0"
                    "END)" % column,
                    #"0",
                    "SUM(CASE TRUE"
                    "       WHEN %s IS NULL THEN 0"
                    "       ELSE 1"
                    "END)" % column,
                )))
            else:
                column = column.to_sql()
                stats.append(", ".join((
                    "NULL",
                    "NULL",
                    "NULL",
                    "NULL",
                    "SUM(CASE TRUE"
                    "       WHEN %s IS NULL THEN 1"
                    "       ELSE 0"
                    "END)" % column,
                    "SUM(CASE TRUE"
                    "       WHEN %s IS NULL THEN 0"
                    "       ELSE 1"
                    "END)" % column,
                )))

        stats_sql = ", ".join(stats)
        cur = self.connection.cursor()
        cur.execute("""SELECT %s FROM "%s" %s""" % (
            stats_sql, self.table_name, where))
        self.connection.commit()
        results = cur.fetchone()
        stats = []
        for i in range(len(columns)):
            stats.append(results[6*i:6*(i+1)])
        return stats

    def _compute_distributions(self, columns=None):
        if columns is not None:
            columns = [self.domain.var_from_domain(col) for col in columns]
        else:
            columns = list(self.domain)
        where = self._construct_where()
        return self._get_distributions(columns, where)

    def _get_distributions(self, columns, where):
        dists = []
        cur = self.connection.cursor()
        for col in columns:
            cur.execute("""
                SELECT %(col)s, COUNT(%(col)s)
                  FROM "%(table)s"
                    %(where)s
              GROUP BY %(col)s
              ORDER BY %(col)s""" %
                        dict(col=col.to_sql(),
                             table=self.table_name,
                             where=where))
            dist = np.array(cur.fetchall())
            if col.var_type == col.VarTypes.Continuous:
                dists.append((dist.T, []))
            else:
                dists.append((dist[:, 1].T, []))
        self.connection.commit()
        return dists

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
