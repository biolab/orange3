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


class ValuesSql(filter.Values):
    def to_sql(self):
        aggregator = " AND " if self.conjunction else " OR "
        sql = aggregator.join(c.to_sql() for c in self.conditions)
        if self.negate:
            sql = 'NOT (%s)' % sql
        return sql


class FilterDiscreteSql(filter.FilterDiscrete):
    def to_sql(self):
        if self.values is not None:
            return "%s IN (%s)" % (self.column, ','.join(self.values))
        else:
            return "%s IS NOT NULL" % self.column
