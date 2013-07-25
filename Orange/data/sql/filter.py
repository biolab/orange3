from .. import filter


class IsDefinedSql(filter.IsDefined):
    def to_sql(self, data):
        columns = set(self.columns) if self.columns is not None else None
        sql = " AND ".join([
            '%s IS NOT NULL' % a.to_sql()
            for i, a in enumerate(data.domain.attributes)
            if self.columns is None or i in columns
        ])
        if self.negate:
            sql = 'NOT (%s)' % sql
        return sql



