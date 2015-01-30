class Transformation:
    def __init__(self, variable):
        self.variable = variable
        self.last_domain = None


class ColumnTransformation(Transformation):
    def __call__(self, data):
        if self.last_domain != data.domain:
            self.last_domain = data.domain
            self.attr_index = data.domain.index(self.variable)
        return self._transform(data.get_column_view(self.attr_index)[0])

    def _transform(self, c):
        raise NotImplementedError("ColumnTransformations must implement _transform.")


class Identity(ColumnTransformation):
    def _transform(self, c):
        return c


class Indicator(ColumnTransformation):
    def __init__(self, variable, value):
        super().__init__(variable)
        self.value = value

    def _transform(self, c):
        return c == self.value


class Indicator_1(ColumnTransformation):
    def __init__(self, variable, value):
        super().__init__(variable)
        self.value = value

    def _transform(self, c):
        return (c == self.value) * 2 - 1


class Normalizer(ColumnTransformation):
    def __init__(self, variable, offset, factor):
        super().__init__(variable)
        self.offset = offset
        self.factor = factor

    def _transform(self, c):
        return (c - self.offset) * self.factor


class Lookup(ColumnTransformation):
    def __init__(self, variable, lookup_table):
        super().__init__(variable)
        self.lookup_table = lookup_table

    def _transform(self, c):
        return self.lookup_table[c]

