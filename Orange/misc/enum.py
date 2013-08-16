## Based on PEP-354, copied and adapted from
## http://stackoverflow.com/questions/36932/

_enum_registry = {}


def enum_value_unpickler(names, value):
    enum_class = _enum_registry.get(names, None) or Enum(*names)
    return enum_class[value]

def Enum(*names):
    """
    Constructs a class with named constants. The returned class serves as a
    namespace, and adds support for getting items and iteration. Values are
    instances of a class derived from `int` that can be printed out in symbolic
    form.

    :param names: names of constants
    :return: class with named constants.

    Typical use is::

        VarTypes = Enum("None", "Discrete", "Continuous", "String")

    in `Orange.data.Variable`, which adds a namespace VarType to type Variable
    to have constants `Orange.data.Variable.VarTypes.None`,
    `Orange.data.Variable.VarTypes.Discrete` and so forth.

    The resulting class has also a method `pull_up(self, cls)` that puts the
    constants into the namespace of the given class `cls`. The following code
    is used (in module scope) to add named constants to class
    `FilterContinuous`::

         Enum("Equal", "NotEqual", "Less", "LessEqual",
              "Greater", "GreaterEqual", "Between", "Outside",
              "IsDefined").pull_up(FilterContinuous)

    With this, the class `FilterContinuous` has constants
    `FilterContinuous.Equal`, `FilterContinuous.NotEqual` and so forth, and not
    `FilterContinuous.SomeNamespaceName.Equal`.
    """
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

    class EnumValue(int):
        EnumType = property(lambda self: EnumType)

        def __repr__(self):
            return str(names[self])

        __str__ = __repr__

        def __reduce__(self):
            return enum_value_unpickler, (names, int(self))

    constants = [None] * len(names)
    for i, each in enumerate(names):
        val = EnumValue(i)
        setattr(EnumClass, each, val)
        constants[i] = val
    constants = tuple(constants)
    EnumType = EnumClass()
    _enum_registry[names] = EnumType
    return EnumType
