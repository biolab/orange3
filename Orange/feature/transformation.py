class Transformation:
    def __init__(self, variable):
        self.variable = variable
        self.last_domain = None


class Identity(Transformation):
    def __call__(self, data):
        if self.last_domain != data.domain:
            self.last_domain = data.domain
            self.attr_index = data.domain.index(self.variable)
        return data.get_column_view(self.attr_index)[0]


class Indicator(Transformation):
    def __init__(self, variable, value):
        super().__init__(variable)
        self.value = value

    def __call__(self, data):
        if self.last_domain != data.domain:
            self.last_domain = data.domain
            self.attr_index = data.domain.index(self.variable)
        return data.get_column_view(self.attr_index)[0] == self.value


class Indicator_1(Transformation):
    def __init__(self, variable, value):
        super().__init__(variable)
        self.value = value

    def __call__(self, data):
        if self.last_domain != data.domain:
            self.last_domain = data.domain
            self.attr_index = data.domain.index(self.variable)
        return (data.get_column_view(self.attr_index)[0] == self.value) * 2 - 1


class Normalizer(Transformation):
    def __init__(self, variable, offset, factor):
        super().__init__(variable)
        self.offset = offset
        self.factor = factor

    def __call__(self, data):
        if self.last_domain != data.domain:
            self.last_domain = data.domain
            self.attr_index = data.domain.index(self.variable)
        return (data.get_column_view(self.attr_index)[0] - self.offset) * self.factor
