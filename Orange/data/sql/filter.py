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


class FilterContinuousSql(filter.FilterContinuous):
    def to_sql(self):
        if self.oper == self.Equal:
            return "%s = %s" % (self.column, self.ref)
        elif self.oper == self.NotEqual:
            return "%s <> %s" % (self.column, self.ref)
        elif self.oper == self.Less:
            return "%s < %s" % (self.column, self.ref)
        elif self.oper == self.LessEqual:
            return "%s <= %s" % (self.column, self.ref)
        elif self.oper == self.Greater:
            return "%s > %s" % (self.column, self.ref)
        elif self.oper == self.GreaterEqual:
            return "%s >= %s" % (self.column, self.ref)
        elif self.oper == self.Between:
            return "%s >= %s AND %s <= %s" % (self.column, self.ref,
                                              self.column, self.max)
        elif self.oper == self.Outside:
            return "%s < %s OR %s > %s" % (self.column, self.ref,
                                           self.column, self.max)
        elif self.oper == self.IsDefined:
            return "%s IS NOT NULL" % self.column
        else:
            raise ValueError("Invalid operator")
