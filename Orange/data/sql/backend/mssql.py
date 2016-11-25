import re
import warnings
from contextlib import contextmanager

import pymssql

from Orange.data import StringVariable, TimeVariable, ContinuousVariable, DiscreteVariable
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import ToSql, BackendError


class PymssqlBackend(Backend):
    display_name = "SQL Server"

    def __init__(self, connection_params):
        connection_params["server"] = connection_params.pop("host", None)

        for key in list(connection_params):
            if connection_params[key] is None:
                del connection_params[key]

        super().__init__(connection_params)
        try:
            self.connection = pymssql.connect(login_timeout=5, **connection_params)
        except pymssql.Error as ex:
            raise BackendError(str(ex)) from ex

    def list_tables_query(self, schema=None):
        return """
        SELECT [TABLE_SCHEMA], [TABLE_NAME]
          FROM information_schema.tables
         WHERE TABLE_TYPE='BASE TABLE'
      ORDER BY [TABLE_NAME]
        """

    def quote_identifier(self, name):
        return "[{}]".format(name)

    def unquote_identifier(self, quoted_name):
        return quoted_name[1:-1]

    def create_sql_query(self, table_name, fields, filters=(),
                         group_by=None, order_by=None, offset=None, limit=None,
                         use_time_sample=None):
        sql = ["SELECT"]
        if limit and not offset:
            sql.extend(["TOP", str(limit)])
        sql.append(', '.join(fields))
        sql.extend(["FROM", table_name])
        if use_time_sample:
            sql.append("TABLESAMPLE system_time(%i)" % use_time_sample)
        if filters:
            sql.extend(["WHERE", " AND ".join(filters)])
        if group_by:
            sql.extend(["GROUP BY", ", ".join(group_by)])

        if offset and not order_by:
            order_by = fields[0].split("AS")[1:]

        if order_by:
            sql.extend(["ORDER BY", ",".join(order_by)])
        if offset:
            sql.extend(["OFFSET", str(offset), "ROWS"])
            if limit:
                sql.extend(["FETCH FIRST", str(limit), "ROWS ONLY"])

        return " ".join(sql)

    @contextmanager
    def execute_sql_query(self, query, params=()):
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, *params)
                yield cur
        except pymssql.Error as ex:
            raise BackendError(str(ex)) from ex

    def create_variable(self, field_name, field_metadata, type_hints, inspect_table=None):
        if field_name in type_hints:
            var = type_hints[field_name]
        else:
            var = self._guess_variable(field_name, field_metadata,
                                       inspect_table)

        field_name_q = self.quote_identifier(field_name)
        if var.is_continuous:
            if isinstance(var, TimeVariable):
                var.to_sql = ToSql("DATEDIFF(s, '1970-01-01 00:00:00', {})".format(field_name_q))
            else:
                var.to_sql = ToSql(field_name_q)
        else:  # discrete or string
            var.to_sql = ToSql(field_name_q)
        return var

    def _guess_variable(self, field_name, field_metadata, inspect_table):
        from pymssql import STRING, NUMBER, DATETIME, DECIMAL

        type_code, *_ = field_metadata

        if type_code in (NUMBER, DECIMAL):
            return ContinuousVariable(field_name)

        if type_code == DATETIME:
            tv = TimeVariable(field_name)
            tv.have_date = True
            tv.have_time = True
            return tv

        if type_code == STRING:
            if inspect_table:
                values = self.get_distinct_values(field_name, inspect_table)
                if values:
                    return DiscreteVariable(field_name, values)

        return StringVariable(field_name)

    EST_ROWS_RE = re.compile(r'StatementEstRows="(\d+)"')

    def count_approx(self, query):
        try:
            with self.connection.cursor() as cur:
                cur.execute("SET SHOWPLAN_XML ON")
                try:
                    cur.execute(query)
                    result = cur.fetchone()
                    return int(self.EST_ROWS_RE.search(result[0]).group(1))
                finally:
                    cur.execute("SET SHOWPLAN_XML OFF")
        except pymssql.Error as ex:
            if "SHOWPLAN permission denied" in str(ex):
                warnings.warn("SHOWPLAN permission denied, count approximates will not be used")
                return None
            raise BackendError(str(ex)) from ex
