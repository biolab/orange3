from .. import filter


class IsDefinedSql(filter.IsDefined):
    def to_sql(self):
        sql = " AND ".join([
            '%s IS NOT NULL' % column
            for column in self.columns
        ])
        if self.negate:
            sql = 'NOT (%s)' % sql
        return sql


class SameValueSql(filter.SameValue):
    def to_sql(self):
        if self.value is None:
            sql = '%s IS NULL' % self.column
        else:
            sql = "%s = %s" % (self.column, self.value)
        if self.negate:
            sql = 'NOT (%s)' % sql
        return sql
