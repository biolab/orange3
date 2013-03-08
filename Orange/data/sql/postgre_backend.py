import psycopg2


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
        cur.execute("select column_name, data_type "
                    "  from INFORMATION_SCHEMA.COLUMNS "
                    " where table_name = %s;", (self.table_name,))
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
        cur.execute("SELECT DISTINCT %s FROM %s ORDER BY %s" %
                    (field_name, self.table_name, field_name))
        return tuple(x[0] for x in cur.fetchall())

    def _get_nrows(self, cur):
        cur.execute("SELECT COUNT(*) FROM %s" % self.table_name)
        return cur.fetchone()[0]

    def query(self, fields=None, filter=None, row_filter=None):
        param_fields = fields
        if param_fields is not None:
            fields = []
            for field in fields:
                if field.get_value_from is not None:
                    transformer = field.get_value_from(None)
                    if not isinstance(transformer, str):
                        raise ValueError("cannot use transformers "
                                         "that do not return strings "
                                         "with sql backend")
                    field_str = '(%s) AS %s' % (transformer, field.name)
                else:
                    field_str = field.name
                fields.append(field_str)
            if not fields:
                raise ValueError("No fields selected.")
        else:
            fields = ["*"]


class TableInfo(object):
    def __init__(self, fields, nrows):
        self.fields = fields
        self.nrows = nrows
        self.field_names = tuple(name for name, _, _ in fields)
        self.values = {
            name: values
            for name, _, values in fields
        }
