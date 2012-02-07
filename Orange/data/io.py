"""\
*****************
Data I/O (``io``)
*****************

Import/Export
=============

This module contains the functions for importing and exporting Orange
data tables from/to different file formats. This works by associating
a filename extension with a set of loading/saving functions using
:obj:`register_file_type`.

Support for some formats is already implemented:  
    
    - Weka `.arff` format
    - C4.5 `.data/.names` format
    - LibSVM data format
    - R `.R` data frame source (export only)


.. function:: register_file_type(format_name, load_func, save_func, extension)

    Register the ``save_func``, ``load_func`` pair for the 
    ``format_name``. The format is identified by the ``extension``.
    
    :param format_name: the name of the format.
    :type format_name: str
    
    :param load_func: a function used for loading the data (see 
        :ref:`custom-formats` for details)
    :type load_func: function
    
    :param save_func: a function used for saving the data (see 
        :ref:`custom-formats` for details)
    :type save_func: function
    
    :param extension: the file extension associated with this format 
        (e.g. '.myformat'). This can be a list of extension if the 
        format uses multiple extensions (for instance the 
        `.data` and `.names` file pairs in the C4.5 format)
    
    Example from the :obj:`~Orange.data.io` module that registers the Weka .arff 
    format ::
        
        register_file_type("Weka", load_ARFF, to_ARFF, ".arff")
        
``load_func`` or ``save_func`` can be None, indicating that the
corresponding functionality is not supported.
 
Loading and saving from/to custom formats then works the same way as
the standard Orange `.tab` file but with a different filename
extension. ::

    >>> import Orange
    >>> data = Orange.data.Table("iris.arff")
    >>> data.save("Copy of iris.arff")
  
    

.. _custom-formats:

Implementing custom import/export functions.
--------------------------------------------

The signature for the custom load functions should be

``load_myformat(filename, create_new_on=Orange.feature.Descriptor.MakeStatus.NoRecognizedValues, **kwargs)``
    
When constructing variables :obj:`Orange.feature.Descriptor.make` should 
be used with the ``create_new_on`` parameter. 
:obj:`~Orange.feature.Descriptor.make` will return an attribute and the 
status of the variable, telling whether a new attribute was created 
or the old one reused and why (see :mod:`Orange.feature`). 
Additional keyword arguments can be provided in the call to 
:obj:`~Orange.data.Table` constructor. These will be passed in the 
``**kwargs``. 
The function should return the build :obj:`~Orange.data.Table` object.
For examples see the source code for the ``Orange.data.io`` module

The save function is easier to implement.

``save_myformat(filename, table, **kwargs)``

Similar as above the ``**kwargs`` contains any additional arguments
:obj:`~Orange.data.Table.save`.
  
"""
import os

import Orange
import Orange.feature
import Orange.misc
from Orange.core import \
     BasketFeeder, FileExampleGenerator, BasketExampleGenerator, \
     C45ExampleGenerator, TabDelimExampleGenerator, \
     registerFileType as register_file_type

from Orange.data import variable
from Orange.feature import Descriptor
MakeStatus = Orange.feature.Descriptor.MakeStatus
make = Orange.feature.Descriptor.make

def loadARFF(filename, create_on_new=MakeStatus.Incompatible, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in Weka ARFF format
       if there exists no .xml file with the same name. If it does, a multi-label
       dataset is read and returned.
    """
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    if os.path.exists(filename + ".xml") and os.path.exists(filename + ".arff"):
        xml_name = filename + ".xml"
        arff_name = filename + ".arff"
        return Orange.multilabel.mulan.trans_mulan_data(xml_name, arff_name, create_on_new)
    else:
        return loadARFF_Weka(filename, create_on_new)

def loadARFF_Weka(filename, create_on_new=MakeStatus.Incompatible, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in Weka ARFF format"""
    if not os.path.exists(filename) and os.path.exists(filename + ".arff"):
        filename = filename + ".arff"
    f = open(filename, 'r')

    attributes = []
    attributeLoadStatus = []

    name = ''
    state = 0 # header
    data = []
    for l in f.readlines():
        l = l.rstrip("\n") # strip \n
        l = l.replace('\t', ' ') # get rid of tabs
        x = l.split('%')[0] # strip comments
        if len(x.strip()) == 0:
            continue
        if state == 0 and x[0] != '@':
            print "ARFF import ignoring:", x
        if state == 1:
            if x[0] == '{':#sparse data format, begin with '{', ends with '}'
                r = [None] * len(attributes)
                dd = x[1:-1]
                dd = dd.split(',')
                for xs in dd:
                    y = xs.split(" ")
                    if len(y) <> 2:
                        raise ValueError("the format of the data is error")
                    r[int(y[0])] = y[1]
                data.append(r)
            else:#normal data format, split by ','
                dd = x.split(',')
                r = []
                for xs in dd:
                    y = xs.strip(" ")
                    if len(y) > 0:
                        if y[0] == "'" or y[0] == '"':
                            r.append(xs.strip("'\""))
                        else:
                            ns = xs.split()
                            for ls in ns:
                                if len(ls) > 0:
                                    r.append(ls)
                    else:
                        r.append('?')
                data.append(r[:len(attributes)])
        else:
            y = []
            for cy in x.split(' '):
                if len(cy) > 0:
                    y.append(cy)
            if str.lower(y[0][1:]) == 'data':
                state = 1
            elif str.lower(y[0][1:]) == 'relation':
                name = str.strip(y[1])
            elif str.lower(y[0][1:]) == 'attribute':
                if y[1][0] == "'":
                    atn = y[1].strip("' ")
                    idx = 1
                    while y[idx][-1] != "'":
                        idx += 1
                        atn += ' ' + y[idx]
                    atn = atn.strip("' ")
                else:
                    atn = y[1]
                z = x.split('{')
                w = z[-1].split('}')
                if len(z) > 1 and len(w) > 1:
                    # there is a list of values
                    vals = []
                    for y in w[0].split(','):
                        sy = y.strip(" '\"")
                        if len(sy) > 0:
                            vals.append(sy)
                    a, s = make(atn, Orange.data.Type.Discrete, vals, [], create_on_new)
                else:
                    # real...
                    a, s = make(atn, Orange.data.Type.Continuous, [], [], create_on_new)

                attributes.append(a)
                attributeLoadStatus.append(s)
    # generate the domain
    d = Orange.data.Domain(attributes)
    lex = []
    for dd in data:
        e = Orange.data.Instance(d, dd)
        lex.append(e)
    t = Orange.data.Table(d, lex)
    t.name = name

    #if hasattr(t, "attribute_load_status"):
    t.setattr("attribute_load_status", attributeLoadStatus)
    return t
loadARFF = Orange.misc.deprecated_keywords(
{"createOnNew": "create_on_new"}
)(loadARFF)


def toARFF(filename, table, try_numericize=0):
    """Save class:`Orange.data.Table` to file in Weka's ARFF format"""
    t = table
    if filename[-5:] == ".arff":
        filename = filename[:-5]
    #print filename
    f = open(filename + '.arff', 'w')
    f.write('@relation %s\n' % t.domain.classVar.name)
    # attributes
    ats = [i for i in t.domain.attributes]
    ats.append(t.domain.classVar)
    for i in ats:
        real = 1
        if i.varType == 1:
            if try_numericize:
                # try if all values numeric
                for j in i.values:
                    try:
                        x = float(j)
                    except:
                        real = 0 # failed
                        break
            else:
                real = 0
        iname = str(i.name)
        if iname.find(" ") != -1:
            iname = "'%s'" % iname
        if real == 1:
            f.write('@attribute %s real\n' % iname)
        else:
            f.write('@attribute %s { ' % iname)
            x = []
            for j in i.values:
                s = str(j)
                if s.find(" ") == -1:
                    x.append("%s" % s)
                else:
                    x.append("'%s'" % s)
            for j in x[:-1]:
                f.write('%s,' % j)
            f.write('%s }\n' % x[-1])

    # examples
    f.write('@data\n')
    for j in t:
        x = []
        for i in range(len(ats)):
            s = str(j[i])
            if s.find(" ") == -1:
                x.append("%s" % s)
            else:
                x.append("'%s'" % s)
        for i in x[:-1]:
            f.write('%s,' % i)
        f.write('%s\n' % x[-1])

def loadMULAN(filename, create_on_new=MakeStatus.Incompatible, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in Mulan ARFF and XML format"""
    if filename[-4:] == ".xml":
        filename = filename[:-4]
    if os.path.exists(filename + ".xml") and os.path.exists(filename + ".arff"):
        xml_name = filename + ".xml"
        arff_name = filename + ".arff"
        return Orange.multilabel.mulan.trans_mulan_data(xml_name, arff_name)
    else:
        return None
loadARFF = Orange.misc.deprecated_keywords(
{"createOnNew": "create_on_new"}
)(loadARFF)

def toC50(filename, table, try_numericize=0):
    """Save class:`Orange.data.Table` to file in C50 format"""
    t = table
    # export names
#    basename = os.path.basename(filename)
    filename_prefix, ext = os.path.splitext(filename)
    f = open('%s.names' % filename_prefix, 'w')
    f.write('%s.\n\n' % t.domain.class_var.name)
    # attributes
    ats = [i for i in t.domain.attributes]
    ats.append(t.domain.classVar)
    for i in ats:
        real = 1
        # try if real
        if i.varType == Orange.core.VarTypes.Discrete:
            if try_numericize:
                # try if all values numeric
                for j in i.values:
                    try:
                        x = float(j)
                    except Exception:
                        real = 0 # failed
                        break
            else:
                real = 0
        if real == 1:
            f.write('%s: continuous.\n' % i.name)
        else:
            f.write('%s: ' % i.name)
            x = []
            for j in i.values:
                x.append('%s' % j)
            for j in x[:-1]:
                f.write('%s,' % j)
            f.write('%s.\n' % x[-1])
    # examples
    f.close()

    f = open('%s.data' % filename_prefix, 'w')
    for j in t:
        x = []
        for i in range(len(ats)):
            x.append('%s' % j[i])
        for i in x[:-1]:
            f.write('%s,' % i)
        f.write('%s\n' % x[-1])

def toR(filename, t):
    """Save class:`Orange.data.Table` to file in R format"""
    if str.upper(filename[-2:]) == ".R":
        filename = filename[:-2]
    f = open(filename + '.R', 'w')

    atyp = []
    aord = []
    labels = []
    as0 = []
    for a in t.domain.variables:
        as0.append(a)
#    as0.append(t.domain.class_var)
    for a in as0:
        labels.append(str(a.name))
        atyp.append(a.var_type)
        aord.append(a.ordered)

    f.write('data <- data.frame(\n')
    for i in xrange(len(labels)):
        if atyp[i] == 2: # continuous
            f.write('"%s" = c(' % (labels[i]))
            for j in xrange(len(t)):
                if t[j][i].isSpecial():
                    f.write('NA')
                else:
                    f.write(str(t[j][i]))
                if (j == len(t) - 1):
                    f.write(')')
                else:
                    f.write(',')
        elif atyp[i] == 1: # discrete
            if aord[i]: # ordered
                f.write('"%s" = ordered(' % labels[i])
            else:
                f.write('"%s" = factor(' % labels[i])
            f.write('levels=c(')
            for j in xrange(len(as0[i].values)):
                f.write('"x%s"' % (as0[i].values[j]))
                if j == len(as0[i].values) - 1:
                    f.write('),c(')
                else:
                    f.write(',')
            for j in xrange(len(t)):
                if t[j][i].isSpecial():
                    f.write('NA')
                else:
                    f.write('"x%s"' % str(t[j][i]))
                if (j == len(t) - 1):
                    f.write('))')
                else:
                    f.write(',')
        else:
            raise "Unknown attribute type."
        if (i < len(labels) - 1):
            f.write(',\n')
    f.write(')\n')

def toLibSVM(filename, example):
    """Save class:`Orange.data.Table` to file in LibSVM format"""
    import Orange.classification.svm
    Orange.classification.svm.tableToSVMFormat(example, open(filename, "wb"))

def loadLibSVM(filename, create_on_new=MakeStatus.Incompatible, **kwargs):
    """Return class:`Orange.data.Table` containing data from file in LibSVM format"""
    attributeLoadStatus = {}
    def make_float(name):
        attr, s = Orange.feature.Descriptor.make(name, Orange.data.Type.Continuous, [], [], create_on_new)
        attributeLoadStatus[attr] = s
        return attr

    def make_disc(name, unordered):
        attr, s = Orange.feature.Descriptor.make(name, Orange.data.Type.Discrete, [], unordered, create_on_new)
        attributeLoadStatus[attr] = s
        return attr

    data = [line.split() for line in open(filename, "rb").read().splitlines() if line.strip()]
    vars = type("attr", (dict,), {"__missing__": lambda self, key: self.setdefault(key, make_float(key))})()
    item = lambda i, v: (vars[i], vars[i](v))
    values = [dict([item(*val.split(":"))  for val in ex[1:]]) for ex in data]
    classes = [ex[0] for ex in data]
    disc = all(["." not in c for c in classes])
    attributes = sorted(vars.values(), key=lambda var: int(var.name))
    classVar = make_disc("class", sorted(set(classes))) if disc else make_float("target")
    attributeLoadStatus = [attributeLoadStatus[attr] for attr in attributes] + \
                          [attributeLoadStatus[classVar]]
    domain = Orange.data.Domain(attributes, classVar)
    table = Orange.data.Table([Orange.data.Instance(domain, [ex.get(attr, attr("?")) for attr in attributes] + [c]) for ex, c in zip(values, classes)])
    table.attribute_load_status = attributeLoadStatus
    return table
loadLibSVM = Orange.misc.deprecated_keywords(
{"createOnNew": "create_on_new"}
)(loadLibSVM)


"""\
A general CSV file reader.
--------------------------

Currently not yet documented and not registered (needs testing).

"""

def split_escaped_str(str, split_str=" ", escape="\\"):
    res = []
    index = 0
    start = 0
    find_start = 0
    while index != -1:
        index = str.find(split_str, find_start)
        if index != -1 and index > 0:
            if str[index - 1] == escape: # Skip the escaped split_str
                find_start = index + 1
            else:
                res.append(str[start:index])
                start = find_start = index + 1

        elif index == -1:
            res.append(str[start:])
    return res

def is_standard_var_def(cell):
    """Is the cell a standard variable definition (empty, cont, disc, string)
    """
    try:
        var_type(cell)
        return True
    except ValueError, ex:
        return False

def is_var_types_row(row):
    """ Is the row a variable type definition row (as in the orange .tab file)
    """
    return all(map(is_standard_var_def, row))

def var_type(cell):
    """ Return variable type from a variable type definition in cell. 
    """
    if cell in ["c", "continuous"]:
        return variable.Continuous
    elif cell in ["d", "discrete"]:
        return variable.Discrete
    elif cell in ["s", "string"]:
        return variable.String
    elif cell.startswith("pyhton"):
        return variable.Python
    elif cell == "":
        return variable.Descriptor
    elif len(cell.split(",")) > 1:
        return variable.Discrete, cell.split(",")
    else:
        raise ValueError("Unknown variable type definition %r." % cell)

def var_types(row):
    """ Return variable types from row. 
    """
    return map(var_type, row)

def is_var_attributes_row(row):
    """ Is the row an attribute definition row (i.e. the third row in the
    standard orange .tab file format).
    
    """
    return all(map(is_var_attributes_def, row))

def is_var_attributes_def(cell):
    """ Is the cell a standard variable attributes definition. 
    """
    try:
        var_attribute(cell)
        return True
    except ValueError, ex:
        raise
        return False

def _var_attribute_label_parse(cell):
    """ 
    """
    key_value = split_escaped_str(cell, "=")
    if len(key_value) == 2:
        return tuple(key_value)
    else:
        raise ValueError("Invalid attribute label definition %r." % cell)

def var_attribute(cell):
    """ Return variable specifier ("meta" or "class" or None) and attributes
    labels dict. 
    """
    items = split_escaped_str(cell, " ")
    if cell == "":
        return None, {}
    elif items:
        specifier = None
        if items[0] in ["m", "meta"]:
            specifier = "meta"
            items = items[1:]
        elif items[0] == "class":
            specifier = "class"
            items = items[1:]
        return specifier, dict(map(_var_attribute_label_parse, items))
    else:
        raise ValueError("Unknown attribute label definition")

def var_attributes(row):
    """ Return variable specifiers and label definitions for row
    """
    return map(var_attribute, row)


class _var_placeholder(object):
    """ A place holder for an arbitrary variable while it's values are still unknown.
    """
    def __init__(self, name="", values=[]):
        self.name = name
        self.values = set(values)

class _disc_placeholder(_var_placeholder):
    """ A place holder for discrete variables while their values are not yet known.
    """
    pass

def is_val_cont(cell):
    """ Is cell a string representing a real value.
    """
    try:
        float(cell)
        return True
    except ValueError:
        return False

def is_variable_cont(values, n=None, cutoff=0.5):
    """ Is variable with ``values`` in column (``n`` rows) a continuous variable. 
    """
    cont = sum(map(is_val_cont, values)) or 1e-30
    if n is None:
        n = len(values) or 1
    return (float(cont) / n) > cutoff


def is_variable_discrete(values, n=None, cutoff=0.3):
    """ Is variable with ``values`` in column (``n`` rows) a discrete variable. 
    """
    return not is_variable_cont(values, n)

def is_variable_string(values, n=None, cutuff=0.1):
    """ Is variable with ``values`` in column (``n`` rows) a string variable. 
    """
    return False

def load_csv(file, create_new_on=MakeStatus.Incompatible, **kwargs):
    """ Load an Orange.data.Table from s csv file.
    """
    import csv, numpy
    file = as_open_file(file, "rb")
    snifer = csv.Sniffer()
    sample = file.read(5 * 2 ** 20) # max 5MB sample TODO: What if this is not enough. Try with a bigger sample
    dialect = snifer.sniff(sample)
    has_header = snifer.has_header(sample)
    file.seek(0) # Rewind
    reader = csv.reader(file, dialect=dialect)

    header = types = var_attrs = None

#    if not has_header:
#        raise ValueError("No header in the data file.")

    header = reader.next()

    if header:
        # Try to get variable definitions
        types_row = reader.next()
        if is_var_types_row(types_row):
            types = var_types(types_row)

    if types:
        # Try to get the variable attributes
        # (third line in the standard orange tab format).
        labels_row = reader.next()
        if is_var_attributes_row(labels_row):
            var_attrs = var_attributes(labels_row)

    # If definitions not present fill with blanks
    if not types:
        types = [None] * len(header)
    if not var_attrs:
        var_attrs = [None] * len(header)

    # start from the beginning
    file.seek(0)
    reader = csv.reader(file, dialect=dialect)
    for defined in [header, types, var_attrs]:
        if any(defined): # skip definition rows if present in the file
            reader.next()

    variables = []
    undefined_vars = []
    for i, (name, var_t) in enumerate(zip(header, types)):
        if var_t == variable.Discrete:# We do not have values yet.
            variables.append(_disc_placeholder(name))
            undefined_vars.append((i, variables[-1]))
        elif var_t == variable.Continuous:
            variables.append(variable.make(name, Orange.data.Type.Continuous, [], [], create_new_on))
        elif var_t == variable.String:
            variables.append(variable.make(name, Orange.data.Type.String, [], [], create_new_on))
        elif var_t == variable.Python:
            variables.append(variable.Python(name))
        elif isinstance(var_t, tuple):
            var_t, values = var_t
            if var_t == variable.Discrete:
                variables.append(variable.make(name, Orange.data.Type.Discrete, values, [], create_new_on))
            elif var_t == variable.Python:
                raise NotImplementedError()
        elif var_t is None:
            variables.append(_var_placeholder(name))
            undefined_vars.append((i, variables[-1]))

    data = []
    for row in reader:
        data.append(row)
        for ind, var_def in undefined_vars:
            var_def.values.add(row[ind])

    for ind, var_def in undefined_vars:
        values = var_def.values - set(["?", ""]) # TODO: Other unknown strings?
        values = sorted(values)
        if isinstance(var_def, _disc_placeholder):
            variables[ind] = variable.make(var_def.name, Orange.data.Type.Discrete, [], values, create_new_on)
        elif isinstance(var_def, _var_placeholder):
            if is_variable_cont(values):
                variables[ind] = variable.make(var_def.name, Orange.data.Type.Continuous, [], [], create_new_on)
            elif is_variable_discrete(values):
                variables[ind] = variable.make(var_def.name, Orange.data.Type.Discrete, [], values, create_new_on)
            elif is_variable_string(values):
                variables[ind] = variable.make(var_def.name, Orange.data.Type.String, [], [], create_new_on)
            else:
                raise ValueError("Strange column in the data")

    vars = []
    vars_load_status = []
    attribute_load_status = []
    meta_attribute_load_status = {}
    class_var_load_status = []
    for var, status in vars:
        vars.append(var)
        vars_load_status.append(status)

    attributes = []
    class_var = []
    metas = {}
    attribute_indices = []
    variable_indices = []
    class_indices = []
    meta_indices = []
    for i, ((var, status), var_attr) in enumerate(zip(variables, var_attrs)):
        if var_attr:
            flag, attrs = var_attr
            if flag == "class":
                class_var.append(var)
                class_var_load_status.append(status)
                class_indices.append(i)
            elif flag == "meta":
                mid = Orange.data.new_meta_id()
                metas[mid] = var
                meta_attribute_load_status[mid] = status
                meta_indices.append((i, var))
            else:
                attributes.append(var)
                attribute_load_status.append(status)
                attribute_indices.append(i)
            var.attributes.update(attrs)
        else:
            attributes.append(var)
            attribute_load_status.append(status)
            attribute_indices.append(i)

    if len(class_var) > 1:
        raise ValueError("Multiple class variables defined")

    class_var = class_var[0] if class_var else None

    attribute_load_status += class_var_load_status
    variable_indices = attribute_indices + class_indices
    domain = Orange.data.Domain(attributes, class_var)
    domain.add_metas(metas)
    normal = [[row[i] for i in variable_indices] for row in data]
    meta_part = [[row[i] for i, _ in meta_indices] for row in data]
    table = Orange.data.Table(domain, normal)
    for ex, m_part in zip(table, meta_part):
        for (column, var), val in zip(meta_indices, m_part):
            ex[var] = var(val)

    table.metaAttributeLoadStatus = meta_attribute_load_status
    table.attributeLoadStatus = attribute_load_status

    return table

def as_open_file(file, mode="rb"):
    if isinstance(file, basestring):
        file = open(file, mode)
    else: # assuming it is file like with proper mode, could check for write, read
        pass
    return file

def save_csv(file, table, orange_specific=True, **kwargs):
    import csv
    file = as_open_file(file, "wb")
    writer = csv.writer(file, **kwargs)
    attrs = table.domain.attributes
    class_var = table.domain.class_var
    metas = [v for _, v in sorted(table.domain.get_metas().items(),
                                  reverse=True)]
    all_vars = attrs + ([class_var] if class_var else []) + metas
    names = [v.name for v in all_vars]
    writer.writerow(names)

    if orange_specific:
        type_cells = []
        for v in all_vars:
            if isinstance(v, variable.Discrete):
                type_cells.append(",".join(v.values))
            elif isinstance(v, variable.Continuous):
                type_cells.append("continuous")
            elif isinstance(v, variable.String):
                type_cells.append("string")
            elif isinstance(v, variable.Python):
                type_cells.append("python")
            else:
                raise TypeError("Unknown variable type")
        writer.writerow(type_cells)

        var_attr_cells = []
        for spec, var in [("", v) for v in attrs] + \
                         ([("class", class_var)] if class_var else []) + \
                         [("m", v) for v in metas]:

            labels = ["{0}={1}".format(*t) for t in var.attributes.items()] # TODO escape spaces
            var_attr_cells.append(" ".join([spec] if spec else [] + labels))

        writer.writerow(var_attr_cells)

    for instance in table:
        instance = list(instance) + [instance[m] for m in metas]
        writer.writerow(instance)


register_file_type("R", None, toR, ".R")
register_file_type("Weka", loadARFF, toARFF, ".arff")
register_file_type("Mulan", loadMULAN, None, ".xml")
#registerFileType("C50", None, toC50, [".names", ".data", ".test"])
register_file_type("libSVM", loadLibSVM, toLibSVM, ".svm")

registerFileType = Orange.misc.deprecated_function_name(register_file_type)

#__doc__ +=  \
"""\
Search Paths
============

Associate a prefix with a search path for easier data loading.
The paths can be stored in a user specific file.

Example ::

    >>> import Orange, os
    >>> from Orange.data import io
    >>> io.set_search_path("my_datasets",
    ...                     os.path.expanduser("~/Documents/My Datasets"))
    ...                     persistent=True)
    ...
    
    >>> data = Orange.data.Table("mydatasets:dataset1.tab")


.. autofunction:: set_search_path

.. autofunction:: search_paths

.. autofunction:: persistent_search_paths

.. autofunction:: find_file

.. autofunction:: expand_filename

"""
# Non-persistent registered paths
_session_paths = []

import ConfigParser
from ConfigParser import SafeConfigParser

@Orange.misc.lru_cache(maxsize=1)
def persistent_search_paths():
    """ Return a list of persistent registered (prefix, path) pairs
    """

    global_settings_dir = Orange.misc.environ.install_dir
    user_settings_dir = Orange.misc.environ.orange_settings_dir
    parser = SafeConfigParser()
    parser.read([os.path.join(global_settings_dir, "orange-search-paths.cfg"),
                 os.path.join(user_settings_dir, "orange-search-paths.cfg")])
    try:
        items = parser.items("paths")
        defaults = parser.defaults().items()
        items = [i for i in items if i not in defaults]
    except ConfigParser.NoSectionError:
        items = []
    return items

def save_persistent_search_path(prefix, path):
    """ Save the prefix, path pair. If path is None delete the 
    registered prefix.
    
    """
    if isinstance(path, list):
        path = os.path.pathsep.join(path)

    user_settings_dir = Orange.misc.environ.orange_settings_dir
    if not os.path.exists(user_settings_dir):
        try:
            os.makedirs(user_settings_dir)
        except OSError:
            pass

    filename = os.path.join(user_settings_dir, "orange-search-paths.cfg")
    parser = SafeConfigParser()
    parser.read([filename])

    if not parser.has_section("paths"):
        parser.add_section("paths")

    if path is not None:
        parser.set("paths", prefix, path)
    elif parser.has_option("paths", prefix):
        # Remove the registered prefix 
        parser.remove_option("paths", prefix)
    parser.write(open(filename, "wb"))

def search_paths(prefix=None):
    """ Return a list of the registered (prefix, path) pairs.
    """
    persistent_paths = persistent_search_paths()
    paths = _session_paths + persistent_paths
    if prefix is not None:
        for pref, path in paths:
            if pref == prefix:
                return path
        return ""
    else:
        return paths

def set_search_path(prefix, path, persistent=False):
    """ Associate a search path with a prefix.
    
    :param prefix: a prefix
    :type prefix: str
    
    :param path: search path (can also be a list of path strings)
    :type paths: str
    
    :param persistent: if True then the prefix-paths pair will be 
        saved between sessions (default False).
    :type persistent: bool 
    
    """
    global _session_paths

    if isinstance(path, list):
        path = os.path.pathsep.join(path)

    if persistent:
        save_persistent_search_path(prefix, path)
        # Invalidate the persistent_search_paths cache.
        persistent_search_paths.clear()
    else:
        _session_paths.append((prefix, path))


def expand_filename(prefixed_name):
    """ Expand the prefixed filename with the full path.
    ::
        
        >>> from Orange.data import io
        >>> io.set_search_paths("docs", "/Users/aleserjavec/Documents")
        >>> io.expand_filename("docs:my_tab_file.tab")
        '/Users/aleserjavec/Documents/my_tab_file.tab'
        
        
    """
    prefix, filename = prefixed_name.split(":", 1) #TODO: windows drive letters.
    paths = search_paths(prefix)
    if paths:
        path = paths.split(os.path.pathsep, 1)[0]
        return os.path.join(path, filename)
    else:
        raise ValueError("Unknown prefix %r." % prefix)

def find_file(prefixed_name):
    """ Find the prefixed filename and return its full path.
    """
    if not os.path.exists(prefixed_name):
        if ":" not in prefixed_name:
            raise ValueError("Not a prefixed name.")
        prefix, filename = prefixed_name.split(":", 1)
        paths = search_paths(prefix)
        if paths:
            for path in paths.split(os.path.pathsep):
                if os.path.exists(os.path.join(path, filename)):
                    return os.path.join(path, filename)
            raise ValueError("No file %r on prefixed search path %r." % \
                             (filename, paths))
        else:
            raise ValueError("Unknown prefix %r." % prefix)
    else:
        return prefixed_name

