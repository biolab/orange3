import pickle
import re
import sys
import warnings
from typing import Iterable

from ast import literal_eval
from collections import OrderedDict, Counter
from functools import lru_cache
from itertools import chain, repeat
from math import isnan

from os import path, remove
from fnmatch import fnmatch
from glob import glob

import numpy as np

from Orange.data import Table, Domain, Variable, DiscreteVariable, \
    StringVariable, ContinuousVariable, TimeVariable
from Orange.data.io_util import Compression, open_compressed, \
    isnastr, guess_data_type, sanitize_variable
from Orange.util import Registry, flatten, namegen

__all__ = ["FileFormatBase", "Flags"]


class Flags:
    """Parser for column flags (i.e. third header row)"""
    DELIMITER = ' '
    _RE_SPLIT = re.compile(r'(?<!\\)' + DELIMITER).split
    _RE_ATTR_UNQUOTED_STR = re.compile(r'^[a-zA-Z_]').match
    ALL = OrderedDict((
        ('class', 'c'),
        ('ignore', 'i'),
        ('meta', 'm'),
        ('weight', 'w'),
        ('.+?=.*?', ''),  # general key=value attributes
    ))
    _RE_ALL = re.compile(r'^({})$'.format('|'.join(
        filter(None, flatten(ALL.items())))))

    def __init__(self, flags):
        for v in filter(None, self.ALL.values()):
            setattr(self, v, False)
        self.attributes = {}
        for flag in flags or []:
            flag = flag.strip()
            if self._RE_ALL.match(flag):
                if '=' in flag:
                    k, v = flag.split('=', 1)
                    if not Flags._RE_ATTR_UNQUOTED_STR(v):
                        try:
                            v = literal_eval(v)
                        except SyntaxError:
                            # If parsing failed, treat value as string
                            pass
                    self.attributes[k] = v
                else:
                    setattr(self, flag, True)
                    setattr(self, self.ALL.get(flag, ''), True)
            elif flag:
                warnings.warn('Invalid attribute flag \'{}\''.format(flag))

    @staticmethod
    def join(iterable, *args):
        return Flags.DELIMITER.join(i.strip().replace(Flags.DELIMITER,
                                                      '\\' + Flags.DELIMITER)
                                    for i in chain(iterable, args)).lstrip()

    @staticmethod
    def split(s):
        return [i.replace('\\' + Flags.DELIMITER, Flags.DELIMITER)
                for i in Flags._RE_SPLIT(s)]


# Matches discrete specification where all the values are listed, space-separated
_RE_DISCRETE_LIST = re.compile(r'^\s*[^\s]+(\s[^\s]+)+\s*$')
_RE_TYPES = re.compile(r'^\s*({}|{}|)\s*$'.format(
    _RE_DISCRETE_LIST.pattern,
    '|'.join(flatten(getattr(vartype, 'TYPE_HEADERS')
                     for vartype in Variable.registry.values()))
))
_RE_FLAGS = re.compile(r'^\s*( |{}|)*\s*$'.format(
    '|'.join(flatten(filter(None, i) for i in Flags.ALL.items()))
))


class _FileReader:
    @classmethod
    def get_reader(cls, filename):
        """Return reader instance that can be used to read the file

        Parameters
        ----------
        filename : str

        Returns
        -------
        FileFormat
        """
        for ext, reader in cls.readers.items():
            # Skip ambiguous, invalid compression-only extensions added on OSX
            if ext in Compression.all:
                continue
            if fnmatch(path.basename(filename), '*' + ext):
                return reader(filename)

        raise IOError('No readers for file "{}"'.format(filename))

    @classmethod
    def set_table_metadata(cls, filename, table):
        # pylint: disable=bare-except
        if isinstance(filename, str) and path.exists(filename + '.metadata'):
            try:
                with open(filename + '.metadata', 'rb') as f:
                    table.attributes = pickle.load(f)
            # Unpickling throws different exceptions, not just UnpickleError
            except:
                with open(filename + '.metadata', encoding='utf-8') as f:
                    table.attributes = OrderedDict(
                        (k.strip(), v.strip())
                        for k, v in (line.split(":", 1)
                                     for line in f.readlines()))

    @staticmethod
    def parse_headers(data):
        """Return (header rows, rest of data) as discerned from `data`"""

        def is_number(item):
            try:
                float(item)
            except ValueError:
                return False
            return True

        # Second row items are type identifiers
        def header_test2(items):
            return all(map(_RE_TYPES.match, items))

        # Third row items are flags and column attributes (attr=value)
        def header_test3(items):
            return all(map(_RE_FLAGS.match, items))

        data = iter(data)
        header_rows = []

        # Try to parse a three-line header
        lines = []
        try:
            lines.append(list(next(data)))
            lines.append(list(next(data)))
            lines.append(list(next(data)))
        except StopIteration:
            lines, data = [], chain(lines, data)
        if lines:
            l1, l2, l3 = lines
            # Three-line header if line 2 & 3 match (1st line can be anything)
            if header_test2(l2) and header_test3(l3):
                header_rows = [l1, l2, l3]
            else:
                lines, data = [], chain((l1, l2, l3), data)

        # Try to parse a single-line header
        if not header_rows:
            try:
                lines.append(list(next(data)))
            except StopIteration:
                pass
            if lines:
                # Header if none of the values in line 1 parses as a number
                if not all(is_number(i) for i in lines[0]):
                    header_rows = [lines[0]]
                else:
                    data = chain(lines, data)

        return header_rows, data

    @classmethod
    def data_table(cls, data, headers=None):
        """
        Return Orange.data.Table given rows of `headers` (iterable of iterable)
        and rows of `data` (iterable of iterable).

        Basically, the idea of subclasses is to produce those two iterables,
        however they might.

        If `headers` is not provided, the header rows are extracted from
        `data`, assuming they precede it.
        """
        if not headers:
            headers, data = cls.parse_headers(data)

        # Consider various header types (single-row, two-row, three-row, none)
        if len(headers) == 3:
            names, types, flags = map(list, headers)
        else:
            if len(headers) == 1:
                HEADER1_FLAG_SEP = '#'
                # First row format either:
                #   1) delimited column names
                #   2) -||- with type and flags prepended, separated by #,
                #      e.g. d#sex,c#age,cC#IQ
                _flags, names = zip(*[i.split(HEADER1_FLAG_SEP, 1)
                                      if HEADER1_FLAG_SEP in i else ('', i)
                                      for i in headers[0]]
                                    )
                names = list(names)
            elif len(headers) == 2:
                names, _flags = map(list, headers)
            else:
                # Use heuristics for everything
                names, _flags = [], []
            types = [''.join(filter(str.isupper, flag)).lower() for flag in
                     _flags]
            flags = [Flags.join(filter(str.islower, flag)) for flag in _flags]

        # Determine maximum row length
        rowlen = max(map(len, (names, types, flags)))

        strip = False

        def _equal_length(lst):
            nonlocal strip
            if len(lst) > rowlen > 0:
                lst = lst[:rowlen]
                strip = True
            elif len(lst) < rowlen:
                lst.extend([''] * (rowlen - len(lst)))
            return lst

        # Ensure all data is of equal width in a column-contiguous array
        data = [_equal_length([s.strip() for s in row])
                for row in data if any(row)]
        data = np.array(data, dtype=object, order='F')

        if strip:
            warnings.warn("Columns with no headers were removed.")

        # Data may actually be longer than headers were
        try:
            rowlen = data.shape[1]
        except IndexError:
            pass
        else:
            for lst in (names, types, flags):
                _equal_length(lst)

        NAMEGEN = namegen('Feature ', 1)
        Xcols, attrs = [], []
        Mcols, metas = [], []
        Ycols, clses = [], []
        Wcols = []

        # Rename variables if necessary
        # Reusing across files still works if both files have same duplicates
        name_counts = Counter(names)
        del name_counts[""]
        if len(name_counts) != len(names) and name_counts:
            uses = {name: 0 for name, count in name_counts.items() if
                    count > 1}
            for i, name in enumerate(names):
                if name in uses:
                    uses[name] += 1
                    names[i] = "{}_{}".format(name, uses[name])

        namask = np.empty(data.shape[0], dtype=bool)
        # Iterate through the columns
        for col in range(rowlen):
            flag = Flags(Flags.split(flags[col]))
            if flag.i:
                continue

            type_flag = types and types[col].strip()
            try:
                orig_values = data[:, col]
            except IndexError:
                orig_values = np.array([], dtype=object)

            namask = isnastr(orig_values, out=namask)

            coltype_kwargs = {}
            valuemap = None
            values = orig_values

            if type_flag in StringVariable.TYPE_HEADERS:
                coltype = StringVariable
                values = orig_values
            elif type_flag in ContinuousVariable.TYPE_HEADERS:
                coltype = ContinuousVariable
                values = np.empty(data.shape[0], dtype=float)
                try:
                    np.copyto(values, orig_values, casting="unsafe",
                              where=~namask)
                    values[namask] = np.nan
                except ValueError:
                    for row, num in enumerate(orig_values):
                        if not isnastr(num):
                            try:
                                float(num)
                            except ValueError:
                                break
                    raise ValueError(f"Non-continuous value in (1-based) "
                                     f"line {row + len(headers) + 1}, "
                                     f"column {col + 1}")

            elif type_flag in TimeVariable.TYPE_HEADERS:
                coltype = TimeVariable
                values = np.where(namask, "", orig_values)
            elif (type_flag in DiscreteVariable.TYPE_HEADERS or
                  _RE_DISCRETE_LIST.match(type_flag)):
                coltype = DiscreteVariable
                orig_values = values = np.where(namask, "", orig_values)
                if _RE_DISCRETE_LIST.match(type_flag):
                    valuemap = Flags.split(type_flag)
                    coltype_kwargs.update(ordered=True)
                else:
                    valuemap = sorted(set(orig_values) - {""})
            else:
                # No known type specified, use heuristics
                valuemap, values, coltype = guess_data_type(orig_values,
                                                            namask)

            if flag.m or coltype is StringVariable:
                append_to = (Mcols, metas)
            elif flag.w:
                append_to = (Wcols, None)
            elif flag.c:
                append_to = (Ycols, clses)
            else:
                append_to = (Xcols, attrs)

            cols, domain_vars = append_to

            if domain_vars is not None:
                var_name = names and names[col]
                if not var_name:
                    var_name = next(NAMEGEN)

                values, var = sanitize_variable(
                    valuemap, values, orig_values, coltype, coltype_kwargs,
                    name=var_name)
            else:
                var = None
            if domain_vars is not None:
                var.attributes.update(flag.attributes)
                domain_vars.append(var)

            if isinstance(values, np.ndarray) and not values.flags.owndata:
                values = values.copy()  # might view `data` (string columns)
            cols.append(values)

            try:
                # allow gc to reclaim memory used by string values
                data[:, col] = None
            except IndexError:
                pass

        domain = Domain(attrs, clses, metas)

        if not data.size:
            return Table.from_domain(domain, 0)

        X = Y = M = W = None
        if Xcols:
            X = np.c_[tuple(Xcols)]
            assert X.dtype == np.float_
        else:
            X = np.empty((data.shape[0], 0), dtype=np.float_)
        if Ycols:
            Y = np.c_[tuple(Ycols)]
            assert Y.dtype == np.float_
        if Mcols:
            M = np.c_[tuple(Mcols)].astype(object)
        if Wcols:
            W = np.c_[tuple(Wcols)].astype(float)

        table = Table.from_numpy(domain, X, Y, M, W)
        return table


class _FileWriter:
    @classmethod
    def write(cls, filename, data, with_annotations=True):
        if cls.OPTIONAL_TYPE_ANNOTATIONS:
            return cls.write_file(filename, data, with_annotations)
        else:
            return cls.write_file(filename, data)

    @classmethod
    def write_table_metadata(cls, filename, data):
        def write_file(fn):
            if all(isinstance(key, str) and isinstance(value, str)
                   for key, value in data.attributes.items()):
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write("\n".join("{}: {}".format(*kv)
                                      for kv in data.attributes.items()))
            else:
                with open(fn, 'wb') as f:
                    pickle.dump(data.attributes, f, pickle.HIGHEST_PROTOCOL)

        if isinstance(filename, str):
            metafile = filename + '.metadata'
            if getattr(data, 'attributes', None):
                write_file(metafile)
            elif path.exists(metafile):
                remove(metafile)

    @staticmethod
    def header_names(data):
        return ['weights'] * data.has_weights() + \
               [v.name for v in chain(data.domain.attributes,
                                      data.domain.class_vars,
                                      data.domain.metas)]

    @staticmethod
    def header_types(data):
        def _vartype(var):
            if var.is_continuous or var.is_string:
                return var.TYPE_HEADERS[0]
            elif var.is_discrete:
                return Flags.join(var.values) if var.ordered else \
                    var.TYPE_HEADERS[0]
            raise NotImplementedError

        return ['continuous'] * data.has_weights() + \
               [_vartype(v) for v in chain(data.domain.attributes,
                                           data.domain.class_vars,
                                           data.domain.metas)]

    @staticmethod
    def header_flags(data):
        return list(chain(
            ['weight'] * data.has_weights(),
            (Flags.join([flag], *('{}={}'.format(*a) for a in
                                  sorted(var.attributes.items())))
             for flag, var in chain(zip(repeat(''), data.domain.attributes),
                                    zip(repeat('class'),
                                        data.domain.class_vars),
                                    zip(repeat('meta'), data.domain.metas)))))

    @classmethod
    def write_headers(cls, write, data, with_annotations=True):
        """`write` is a callback that accepts an iterable"""
        write(cls.header_names(data))
        if with_annotations:
            write(cls.header_types(data))
            write(cls.header_flags(data))

    @classmethod
    def formatter(cls, var):
        # type: (Variable) -> Callable[[Variable], Any]
        # Return a column 'formatter' function. The function must return
        # something that `write` knows how to write
        if var.is_time:
            return var.repr_val
        elif var.is_continuous:
            return lambda value: "" if isnan(value) else value
        elif var.is_discrete:
            return lambda value: "" if isnan(value) else var.values[int(value)]
        elif var.is_string:
            return lambda value: value
        else:
            return var.repr_val

    @classmethod
    def write_data(cls, write, data):
        """`write` is a callback that accepts an iterable"""
        vars_ = list(
            chain((ContinuousVariable('_w'),) if data.has_weights() else (),
                  data.domain.attributes,
                  data.domain.class_vars,
                  data.domain.metas))

        formatters = [cls.formatter(v) for v in vars_]
        for row in zip(data.W if data.W.ndim > 1 else data.W[:, np.newaxis],
                       data.X,
                       data.Y if data.Y.ndim > 1 else data.Y[:, np.newaxis],
                       data.metas):
            write([fmt(v) for fmt, v in zip(formatters, flatten(row))])


class _FileFormatMeta(Registry):
    def __new__(mcs, name, bases, attrs):
        newcls = super().__new__(mcs, name, bases, attrs)

        # Optionally add compressed versions of extensions as supported
        if getattr(newcls, 'SUPPORT_COMPRESSED', False):
            new_extensions = list(getattr(newcls, 'EXTENSIONS', ()))
            for compression in Compression.all:
                for ext in newcls.EXTENSIONS:
                    new_extensions.append(ext + compression)
                if sys.platform in ('darwin', 'win32'):
                    # OSX file dialog doesn't support filtering on double
                    # extensions (e.g. .csv.gz)
                    # https://bugreports.qt.io/browse/QTBUG-38303
                    # This is just here for OWFile that gets QFileDialog
                    # filters from FileFormat.readers.keys()
                    # EDIT: Windows exhibit similar problems:
                    # while .tab.gz works, .tab.xz and .tab.bz2 do not!
                    new_extensions.append(compression)
            newcls.EXTENSIONS = tuple(new_extensions)

        return newcls

    @property
    def formats(cls):
        return cls.registry.values()

    @lru_cache(5)
    def _ext_to_attr_if_attr2(cls, attr, attr2):
        """
        Return ``{ext: `attr`, ...}`` dict if ``cls`` has `attr2`.
        If `attr` is '', return ``{ext: cls, ...}`` instead.

        If there are multiple formats for an extension, return a format
        with the lowest priority.
        """
        formats = OrderedDict()
        for format_ in sorted(cls.registry.values(), key=lambda x: x.PRIORITY):
            if not hasattr(format_, attr2):
                continue
            for ext in getattr(format_, 'EXTENSIONS', []):
                # Only adds if not yet registered
                formats.setdefault(ext, getattr(format_, attr, format_))
        return formats

    @property
    def names(cls):
        return cls._ext_to_attr_if_attr2('DESCRIPTION', '__class__')

    @property
    def writers(cls):
        return cls._ext_to_attr_if_attr2('', 'write_file')

    @property
    def readers(cls):
        return cls._ext_to_attr_if_attr2('', 'read')

    @property
    def img_writers(cls):
        warnings.warn(
            f"'{__name__}.FileFormat.img_writers' is no longer used and "
            "will be removed. Please use "
            "'Orange.widgets.io.FileFormat.img_writers' instead.",
            DeprecationWarning, stacklevel=2
        )
        return cls._ext_to_attr_if_attr2('', 'write_image')

    @property
    def graph_writers(cls):
        return cls._ext_to_attr_if_attr2('', 'write_graph')


class FileFormatBase(_FileReader, _FileWriter, metaclass=_FileFormatMeta):
    # Priority when multiple formats support the same extension. Also
    # the sort order in file open/save combo boxes. Lower is better.
    PRIORITY = 10000
    OPTIONAL_TYPE_ANNOTATIONS = False

    @classmethod
    def locate(cls, filename, search_dirs=('.',)):
        """Locate a file with given filename that can be opened by one
        of the available readers.

        Parameters
        ----------
        filename : str
        search_dirs : Iterable[str]

        Returns
        -------
        str
            Absolute path to the file
        """
        if path.exists(filename):
            return filename

        for directory in search_dirs:
            absolute_filename = path.join(directory, filename)
            if path.exists(absolute_filename):
                break
            for ext in cls.readers:
                if fnmatch(path.basename(filename), '*' + ext):
                    break
                # glob uses fnmatch internally
                matching_files = glob(absolute_filename + ext)
                if matching_files:
                    absolute_filename = matching_files[0]
                    break
            if path.exists(absolute_filename):
                break
        else:
            absolute_filename = ""

        if not path.exists(absolute_filename):
            raise IOError('File "{}" was not found.'.format(filename))

        return absolute_filename

    @staticmethod
    def open(filename, *args, **kwargs):
        """
        Format handlers can use this method instead of the builtin ``open()``
        to transparently (de)compress files if requested (according to
        `filename` extension). Set ``SUPPORT_COMPRESSED=True`` if you use this.
        """
        return open_compressed(filename, *args, **kwargs)

    @classmethod
    def qualified_name(cls):
        return cls.__module__ + '.' + cls.__name__
