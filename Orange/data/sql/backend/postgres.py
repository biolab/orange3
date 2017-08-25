import logging
import re
import warnings
from contextlib import contextmanager
from time import time

from psycopg2 import Error, OperationalError
from psycopg2.pool import ThreadedConnectionPool

from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable
from Orange.data.sql.backend.base import Backend, ToSql, BackendError

log = logging.getLogger(__name__)

EXTENSIONS = ('tsm_system_time', 'quantile')


class Psycopg2Backend(Backend):
    """Backend for accessing data stored in a Postgres database
    """

    display_name = "PostgreSQL"
    connection_pool = None
    auto_create_extensions = True

    def __init__(self, connection_params):
        super().__init__(connection_params)

        if self.connection_pool is None:
            self._create_connection_pool()

        if self.auto_create_extensions:
            self._create_extensions()

    def _create_connection_pool(self):
        try:
            self.connection_pool = ThreadedConnectionPool(
                1, 16, **self.connection_params)
        except Error as ex:
            raise BackendError(str(ex)) from ex

    def _create_extensions(self):
        for ext in EXTENSIONS:
            try:
                query = "CREATE EXTENSION IF NOT EXISTS {}".format(ext)
                with self.execute_sql_query(query):
                    pass
            except OperationalError:
                warnings.warn("Database is missing extension {}".format(ext))

    def create_sql_query(self, table_name, fields, filters=(),
                         group_by=None, order_by=None,
                         offset=None, limit=None,
                         use_time_sample=None):
        sql = ["SELECT", ', '.join(fields),
               "FROM", table_name]
        if use_time_sample is not None:
            sql.append("TABLESAMPLE system_time(%i)" % use_time_sample)
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
        return " ".join(sql)

    @contextmanager
    def execute_sql_query(self, query, params=None):
        connection = self.connection_pool.getconn()
        cur = connection.cursor()
        try:
            utfquery = cur.mogrify(query, params).decode('utf-8')
            log.debug("Executing: %s", utfquery)
            t = time()
            cur.execute(query, params)
            yield cur
            log.info("%.2f ms: %s", 1000 * (time() - t), utfquery)
        except Error as ex:
            raise BackendError(str(ex)) from ex
        finally:
            connection.commit()
            self.connection_pool.putconn(connection)

    def quote_identifier(self, name):
        return '"%s"' % name

    def unquote_identifier(self, quoted_name):
        if quoted_name.startswith('"'):
            return quoted_name[1:len(quoted_name) - 1]
        else:
            return quoted_name

    def list_tables_query(self, schema=None):
        if schema:
            schema_clause = "AND n.nspname = '{}'".format(schema)
        else:
            schema_clause = "AND pg_catalog.pg_table_is_visible(c.oid)"
        return """SELECT n.nspname as "Schema",
                          c.relname AS "Name"
                       FROM pg_catalog.pg_class c
                  LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                      WHERE c.relkind IN ('r','v','m','S','f','')
                        AND n.nspname <> 'pg_catalog'
                        AND n.nspname <> 'information_schema'
                        AND n.nspname !~ '^pg_toast'
                        {}
                        AND NOT c.relname LIKE '\\_\\_%'
                   ORDER BY 1;""".format(schema_clause)

    def create_variable(self, field_name, field_metadata,
                        type_hints, inspect_table=None):
        if field_name in type_hints:
            var = type_hints[field_name]
        else:
            var = self._guess_variable(field_name, field_metadata,
                                       inspect_table)

        field_name_q = self.quote_identifier(field_name)
        if var.is_continuous:
            if isinstance(var, TimeVariable):
                var.to_sql = ToSql("extract(epoch from {})"
                                   .format(field_name_q))
            else:
                var.to_sql = ToSql("({})::double precision"
                                   .format(field_name_q))
        else:  # discrete or string
            var.to_sql = ToSql("({})::text"
                               .format(field_name_q))
        return var

    def _guess_variable(self, field_name, field_metadata, inspect_table):
        type_code = field_metadata[0]

        FLOATISH_TYPES = (700, 701, 1700)  # real, float8, numeric
        INT_TYPES = (20, 21, 23)  # bigint, int, smallint
        CHAR_TYPES = (25, 1042, 1043,)  # text, char, varchar
        BOOLEAN_TYPES = (16,)  # bool
        DATE_TYPES = (1082, 1114, 1184, )  # date, timestamp, timestamptz
        # time, timestamp, timestamptz, timetz
        TIME_TYPES = (1083, 1114, 1184, 1266,)

        if type_code in FLOATISH_TYPES:
            return ContinuousVariable.make(field_name)

        if type_code in TIME_TYPES + DATE_TYPES:
            tv = TimeVariable.make(field_name)
            tv.have_date |= type_code in DATE_TYPES
            tv.have_time |= type_code in TIME_TYPES
            return tv

        if type_code in INT_TYPES:  # bigint, int, smallint
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable.make(field_name, values)
            return ContinuousVariable.make(field_name)

        if type_code in BOOLEAN_TYPES:
            return DiscreteVariable.make(field_name, ['false', 'true'])

        if type_code in CHAR_TYPES:
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable.make(field_name, values)

        return StringVariable.make(field_name)

    def count_approx(self, query):
        sql = "EXPLAIN " + query
        with self.execute_sql_query(sql) as cur:
            s = ''.join(row[0] for row in cur.fetchall())
        return int(re.findall(r'rows=(\d*)', s)[0])

    def __getstate__(self):
        # Drop connection_pool from state as it cannot be pickled
        state = dict(self.__dict__)
        state.pop('connection_pool', None)
        return state

    def __setstate__(self, state):
        # Create a new connection pool if none exists
        self.__dict__.update(state)
        if self.connection_pool is None:
            self._create_connection_pool()
