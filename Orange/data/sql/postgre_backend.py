import math
import psycopg2
import numpy as np

class PostgreBackend(object):
    def connect(self,
                database,
                table,
                hostname=None,
                username=None,
                password=None):
        self.connection = psycopg2.connect(
            host=hostname,
            user=username,
            password=password,
            database=database,
        )
        self.table_name = table
        self.table_info = self._get_table_info()


    def _get_table_info(self):
        cur = self.connection.cursor()
        return TableInfo(
            fields=self._get_field_list(cur),
            nrows=self._get_nrows(cur),
        )

    def _get_field_list(self, cur):
        cur.execute("""
            SELECT column_name, data_type
              FROM INFORMATION_SCHEMA.COLUMNS
             WHERE table_name = %s;""", (self.table_name,))
        self.connection.commit()
        return tuple(
            (fname, ftype, self._get_field_values(fname, ftype, cur))
            for fname, ftype in cur.fetchall()
        )

    def _get_field_values(self, field_name, field_type, cur):
        if 'double' in field_type:
            return ()
        elif 'char' in field_type:
            return self._get_distinct_values(field_name, cur)

    def _get_distinct_values(self, field_name, cur):
        cur.execute("""SELECT DISTINCT %s FROM "%s" ORDER BY %s LIMIT 21""" %
                    (field_name, self.table_name, field_name))
        self.connection.commit()
        values = cur.fetchall()

        if len(values) > 20:
            return ()
        else:
            return tuple(x[0] for x in values)

    def _get_nrows(self, cur):
        cur.execute("""SELECT COUNT(*) FROM "%s" """ % self.table_name)
        self.connection.commit()
        return cur.fetchone()[0]

    def query(self, attributes=None, filters=None, rows=None):
        if attributes is not None:
            fields = []
            for attr in attributes:
                if attr.get_value_from is not None:
                    field_src = attr.get_value_from(None)
                    if not isinstance(field_src, str):
                        raise ValueError("cannot use ordinary attributes "
                                         "with sql backend")
                    field_str = '(%s) AS %s' % (field_src, attr.name)
                else:
                    field_str = attr.name
                fields.append(field_str)
            if not fields:
                raise ValueError("No fields selected.")
        else:
            fields = ["*"]

        if filters is not None:
            pass

        sql = """SELECT %s FROM "%s" """ % (', '.join(fields), self.table_name)
        if rows is not None:
            if isinstance(rows, slice):
                start = rows.start or 0
                stop = rows.stop or self.table_info.nrows
                size = stop - start
            else:
                rows = list(rows)
                start, stop = min(rows), max(rows)
                size = stop - start + 1
            sql += " OFFSET %d LIMIT %d" % (start, size)
        cur = self.connection.cursor()
        cur.execute(sql)
        self.connection.commit()
        return cur.fetchall()

    def stats(self, columns):
        stats = []
        for column in columns:
            if column.var_type == column.VarTypes.Continuous:
                column = column.get_value_from()
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
                column = column.get_value_from()
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
        cur.execute("""SELECT %s FROM "%s" """ % (stats_sql, self.table_name))
        self.connection.commit()
        results = cur.fetchone()
        stats = []
        for i in range(len(columns)):
            stats.append(results[6*i:6*(i+1)])
        return stats

    def distributions(self, columns):
        dists = []
        cur = self.connection.cursor()
        for col in columns:
            cur.execute("""
                SELECT %(col)s, COUNT(%(col)s)
                  FROM "%(table)s"
              GROUP BY %(col)s
              ORDER BY %(col)s""" %
                        dict(col=col.get_value_from(), table=self.table_name))
            dist = np.array(cur.fetchall())
            if col.var_type == col.VarTypes.Continuous:
                dists.append((dist.T, []))
            else:
                dists.append((dist[:, 1].T, []))
        self.connection.commit()
        return dists





class TableInfo(object):
    def __init__(self, fields, nrows):
        self.fields = fields
        self.nrows = nrows
        self.field_names = tuple(name for name, _, _ in fields)
        self.values = {
            name: values
            for name, _, values in fields
        }
