from .. import filter


class IsDefinedSql(filter.IsDefined):
    def to_sql(self, data):
        columns = set(self.columns) if self.columns is not None else None
        return " AND ".join([
            '%s IS NOT NULL' % a.name
            for i, a in enumerate(data.domain.attributes)
            if self.columns is None or i in columns
        ])



