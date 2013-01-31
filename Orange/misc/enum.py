## Based on PEP-354, copied and adapted from
## http://stackoverflow.com/questions/36932/

_enum_registry = {}


def enum_value_unpickler(names, value):
    enum_class = _enum_registry.get(names, None) or Enum(*names)
    return enum_class[value]

def Enum(*names):
    if names in _enum_registry:
        return _enum_registry[names]

    class EnumClass:
        __slots__ = names

        def __iter__(self):
            return iter(constants)

        def __len__(self):
            return len(constants)

        def __getitem__(self, i):
            return constants[i]

        def __repr__(self):
            return 'Enum' + str(names)

        def __str__(self):
            return 'enum ' + str(constants)

        def add_value(self, value):
            value = EnumValue(len(names))
            setattr(self, value, value)
            constants.append(value)

        def pull_up(self, cls):
            for name, value in zip(names, constants):
                setattr(cls, name, value)

    class EnumValue:
        __slots__ = ('__value', )
        Value = property(lambda self: self.__value)
        EnumType = property(lambda self: EnumType)

        def __init__(self, value):
            self.__value = value

        def __hash__(self):
            return hash(self.__value)

        def __lt__(self, other):
            return self.__value < other.__value

        def __eq__(self, other):
            return self.__value == other.__value

        def __nonzero__(self):
            return bool(self.__value)

        def __repr__(self):
            return str(names[self.__value])

        def __reduce__(self):
            return enum_value_unpickler, (names, self.__value)

    constants = [None] * len(names)
    for i, each in enumerate(names):
        val = EnumValue(i)
        setattr(EnumClass, each, val)
        constants[i] = val
    constants = tuple(constants)
    EnumType = EnumClass()
    _enum_registry[names] = EnumType
    return EnumType