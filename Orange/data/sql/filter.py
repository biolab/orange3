from .. import filter


class IsDefinedSql(filter.IsDefined):
    def __init__(self, columns=None, negate=False, table=None):
        assert table is not None, "Cannot construct sql filter without table"
        if columns is None:
            columns = range(len(table.domain.variables))
        columns = [table.domain.variables[i].to_sql() for i in columns]
        super().__init__(columns, negate)

    def to_sql(self):
        sql = " AND ".join([
            '%s IS NOT NULL' % column
            for column in self.columns
        ])
        if self.negate:
            sql = 'NOT (%s)' % sql
        return sql



