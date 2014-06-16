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
            if self.value is None:
                sql = 'NOT (%s)' % sql
            else:
                sql = '(NOT (%s) OR %s is NULL)' % (sql, self.column)
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
            return "%s <> %s OR %s IS NULL" % (self.column, self.ref, self.column)
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
            return "(%s < %s OR %s > %s)" % (self.column, self.ref,
                                           self.column, self.max)
        elif self.oper == self.IsDefined:
            return "%s IS NOT NULL" % self.column
        else:
            raise ValueError("Invalid operator")


class FilterString(filter.FilterString):
    def to_sql(self):
        if self.oper == self.IsDefined:
            return "%s IS NOT NULL" % self.column
        if self.case_sensitive:
            field = self.column
            value = self.ref
        else:
            field = 'LOWER(%s)' % self.column
            value = self.ref.lower()
        if self.oper == self.Equal:
            return "%s = %s" % (field, quote(value))
        elif self.oper == self.NotEqual:
            return "%s <> %s OR %s IS NULL" % (field, quote(value), field)
        elif self.oper == self.Less:
            return "%s < %s" % (field, quote(value))
        elif self.oper == self.LessEqual:
            return "%s <= %s" % (field, quote(value))
        elif self.oper == self.Greater:
            return "%s > %s" % (field, quote(value))
        elif self.oper == self.GreaterEqual:
            return "%s >= %s" % (field, quote(value))
        elif self.oper == self.Between:
            high = quote(self.max if self.case_sensitive else self.max.lower())
            return "%s >= %s AND %s <= %s" % (field, quote(value), field, high)
        elif self.oper == self.Outside:
            high = quote(self.max if self.case_sensitive else self.max.lower())
            return "(%s < %s OR %s > %s)" % (field, quote(value), field, high)
        elif self.oper == self.Contains:
            return "%s LIKE '%%%s%%'" % (field, value)
        elif self.oper == self.StartsWith:
            return "%s LIKE '%s%%'" % (field, value)
        elif self.oper == self.EndsWith:
            return "%s LIKE '%%%s'" % (field, value)
        else:
            raise ValueError("Invalid operator")


class FilterStringList(filter.FilterStringList):
    def to_sql(self):
        values = self.values
        if not self.case_sensitive:
            values = map(lambda x: x.lower(), values)
            sql = "LOWER(%s) in (%s)"
        else:
            sql = "%s in (%s)"
        return sql % (self.column, ", ".join(map(quote, values)))


def quote(value):
    if isinstance(value, str):
        return "'%s'" % value
    else:
        return value


class CustomFilterSql(filter.Filter):
    def __init__(self, where_sql, negate=False):
        super().__init__(negate=negate)
        self.sql = where_sql

    def to_sql(self):
        if not self.negate:
            return "(" + self.sql + ")"
        else:
            return "NOT (" + self.sql + ")"
