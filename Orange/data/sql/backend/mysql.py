import logging
from contextlib import contextmanager

import mysql.connector as mysql # pylint: disable=import-error

from Orange.data import StringVariable, TimeVariable, ContinuousVariable, DiscreteVariable
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError, ToSql

log = logging.getLogger(__name__)

class PymysqlBackend(Backend):
    """Base class for SqlTable backends. Implementations should define
    all of the methods defined below.

    Parameters
    ----------
    connection_params: dict
        connection params
    """

    display_name = "MySQL"
    connection = None

    def __init__(self, connection_params):
        super().__init__(connection_params)
        connection_params.pop('port')
        self.connection_params = connection_params
        try:
            cursor = self._get_cursor()
            cursor.execute('SELECT VERSION()')
            if cursor.fetchall() is None:
                raise BackendError
            cursor.close()
        except Exception as e:
            raise BackendError(str(e))

    # "meta" methods

    def _get_cursor(self):
        try:
            if self.connection is None:
                self.connection = mysql.connect(**self.connection_params)
            else:
                self.connection.ping(reconnect=True)
            cur = self.connection.cursor()
            cur.execute('SELECT VERSION()')
            if not cur.fetchall():
                raise BackendError("_get_cursor: Connection Failed")
            cur.reset()
            return cur
        except Exception as ex:
            raise BackendError("_get_cursor: "+str(ex))

    def list_tables_query(self, schema=None):
        """Return a list of tuples (schema, table_name)

        Parameters
        ----------
        schema : Optional[str]
            If set, only tables from schema should be listed

        Returns
        -------
        A list of tuples
        """
        if schema:
            schema_clause = "AND table_schema = '{}'".format(schema)
        else:
            with self.execute_sql_query("SELECT CURRENT_USER()") as cur:
                user = cur.fetchone()[0].split('@')[0]
            schema_clause = """TABLE_SCHEMA IN (SELECT DISTINCT(TABLE_SCHEMA)
            FROM information_schema.SCHEMA_PRIVILEGES WHERE GRANTEE LIKE "'{}%'")""".format(user)
        return """SELECT table_schema as "Schema",
                          table_name AS "Name"
                       FROM information_schema.tables
                      WHERE {}""".format(schema_clause)

    def get_fields(self, table_name):
        """Return a list of field names and metadata in the given table

        Parameters
        ----------
        table_name: str

        Returns
        -------
        a list of tuples (field_name, *field_metadata)
        both will be passed to create_variable
        """
        query = "SHOW columns FROM {}".format(table_name)
        with self.execute_sql_query(query) as cur:
            fields = [(f[0], f[1].split("(")[0].upper()) for f in cur.fetchall()]
        return fields

    def create_variable(self, field_name, field_metadata,
                        type_hints, inspect_table=None):
        """Create variable based on field information

        Parameters
        ----------
        field_name : str
            name do the field
        field_metadata : tuple
            data to guess field type from
        type_hints : Domain
            domain with variable templates
        inspect_table : Option[str]
            name of the table to expect the field values or None
            if no inspection is to be performed

        Returns
        -------
        Variable representing the field
        """
        if field_name in type_hints:
            var = type_hints[field_name]
        else:
            var = self._guess_variable(field_name, field_metadata,
                                       inspect_table)

        field_name_q = self.quote_identifier(field_name)
        if var.is_continuous:
            if isinstance(var, TimeVariable):
                var.to_sql = ToSql(field_name_q)
            else:
                var.to_sql = ToSql(field_name_q)
        else:  # discrete or string
            var.to_sql = ToSql(field_name_q)
        return var

    def _guess_variable(self, field_name, field_metadata, inspect_table):
        type_code = field_metadata[0]

        NUMERIC_TYPES = ("FLOAT", "DOUBLE", "DECIMAL")  # real, float8, numeric
        INT_TYPES = ("INT", "TINYINT", "SMALLINT", "MEDIUMINT", "BIGINT")
        DATE_TYPES = ("DATE", "DATETIME", "YEAR")
        TIME_TYPES = ("TIMESTAMP", "TIME")
        CHAR_TYPES = ("CHAR", "ENUM")

        if type_code in NUMERIC_TYPES:
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

        if type_code in CHAR_TYPES:
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                # remove trailing spaces
                values = [v.rstrip() for v in values]
                if values:
                    return DiscreteVariable.make(field_name, values)

        return StringVariable.make(field_name)

    def count_approx(self, query):
        """Return estimated number of rows returned by query.

        Parameters
        ----------
        query : str

        Returns
        -------
        Approximate number of rows
        """
        query = "EXPLAIN " + query
        with self.execute_sql_query(query) as cur:
            return cur.fetchone()[9]

    # query related methods

    def create_sql_query(
            self, table_name, fields, filters=(),
            group_by=None, order_by=None, offset=None, limit=None,
            use_time_sample=None):
        """Construct an sql query using the provided elements.

        Parameters
        ----------
        table_name : str
        fields : List[str]
        filters : List[str]
        group_by: List[str]
        order_by: List[str]
        offset: int
        limit: int
        use_time_sample: int

        Returns
        -------
        string containing sql query
        """
        query = ["SELECT", ", ".join(fields), "FROM", table_name]
        if filters:
            query.extend(["WHERE", " AND ".join(filters)])
        if group_by is not None:
            query.extend(["GROUP BY", ", ".join(group_by)])
        if order_by is not None:
            query.extend(["ORDER BY", ", ".join(order_by)])
        if limit is not None:
            query.extend(["LIMIT", str(limit)])
        if offset is not None:
            if limit is not None:
                query.extend(["OFFSET", str(offset)])
            else:
                query.extend(["LIMIT 18446744073709551615 OFFSET", str(offset)])
        if use_time_sample is not None:
            query.insert(1, "/*+ MAX_EXECUTION_TIME = {} */".format(use_time_sample))
        return " ".join(query)

    @contextmanager
    def execute_sql_query(self, query, params=None):
        """Context manager for execution of sql queries

        Usage:
            ```
            with backend.execute_sql_query("SELECT * FROM foo") as cur:
                cur.fetch_all()
            ```

        Parameters
        ----------
        query : string
            query to be executed
        params: tuple
            parameters to be passed to the query

        Returns
        -------
        yields a cursor that can be used to access the data
        """
        cursor = self._get_cursor()
        if params is not None:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        yield cursor

    def quote_identifier(self, name):
        """Quote identifier name so it can be safely used in queries

        Parameters
        ----------
        name: str
            name of the parameter

        Returns
        -------
        quoted parameter that can be used in sql queries
        """
        return '`{}`'.format(name)

    def unquote_identifier(self, quoted_name):
        """Remove quotes from identifier name
        Used when sql table name is used in where parameter to
        query special tables

        Parameters
        ----------
        quoted_name : str

        Returns
        -------
        unquoted name
        """
        return quoted_name[1:-1]
