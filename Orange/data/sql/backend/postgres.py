import logging
from contextlib import contextmanager
from time import time

from psycopg2.pool import ThreadedConnectionPool

from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable
from Orange.data.sql.backend.base import Backend, ToSql


log = logging.getLogger(__name__)


class Psycopg2Backend(Backend):
    """Backend for accessing data stored in a Postgres database
    """

    connection_pool = None

    def __init__(self, connection_params):
        super().__init__(connection_params)

        if self.connection_pool is None:
            self._create_connection_pool()

    def _create_connection_pool(self):
        self.connection_pool = ThreadedConnectionPool(
            1, 16, **self.connection_params)

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

    def get_fields(self, table_name):
        fields = []
        query = "SELECT * FROM %s LIMIT 0" % table_name
        with self.execute_sql_query(query) as cur:
            for col in cur.description:
                fields.append(col)
        return fields

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
            return ContinuousVariable(field_name)

        if type_code in TIME_TYPES + DATE_TYPES:
            tv = TimeVariable(field_name)
            tv.have_date |= type_code in DATE_TYPES
            tv.have_time |= type_code in TIME_TYPES
            return tv

        if type_code in INT_TYPES:  # bigint, int, smallint
            if inspect_table:
                values = self._get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable(field_name, values)
            return ContinuousVariable(field_name)

        if type_code in BOOLEAN_TYPES:
            return DiscreteVariable(field_name, ['false', 'true'])

        if type_code in CHAR_TYPES:
            if inspect_table:
                values = self._get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable(field_name, values)

        return StringVariable(field_name)

    def _get_distinct_values(self, field_name, table_name):
        field_name_q = self.quote_identifier(field_name)
        sql = " ".join(["SELECT DISTINCT (", field_name_q, ")::text",
                        "FROM", table_name,
                        "WHERE", field_name_q, "IS NOT NULL",
                        "ORDER BY", field_name_q,
                        "LIMIT 21"])
        with self.execute_sql_query(sql) as cur:
            values = cur.fetchall()
        if len(values) > 20:
            return ()
        else:
            return tuple(x[0] for x in values)

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
