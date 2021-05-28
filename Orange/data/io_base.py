import pickle
import re
import sys
import warnings
from typing import Iterable, Optional, Tuple, List, Generator, Callable, Any

from ast import literal_eval
from collections import OrderedDict
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
from Orange.data.util import get_unique_names_duplicates
from Orange.data.variable import VariableMeta
from Orange.misc.collections import natural_sorted
from Orange.util import Registry, flatten, namegen

__all__ = ["FileFormatBase", "Flags", "DataTableMixin", "PICKLE_PROTOCOL"]


PICKLE_PROTOCOL = 4


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
                    # map True and False to booleans
                    if v in ("True", "False"):
                        v = {"True": True, "False": False}[v]
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


class _ColumnProperties:
    def __init__(self, valuemap=None, values=None, orig_values=None,
                 coltype=None, coltype_kwargs=None):
        self.valuemap = valuemap
        self.values = values
        self.orig_values = orig_values
        self.coltype = coltype
        if coltype_kwargs is None:
            self.coltype_kwargs = {}
        else:
            self.coltype_kwargs = dict(coltype_kwargs)


class _TableHeader:
    """
    Contains functions for table header construction (and its data).
    """
    HEADER1_FLAG_SEP = '#'

    def __init__(self, headers: List):
        """
        Parameters
        ----------
        headers: List
            Header rows, to be used for constructing domain.
        """
        names, types, flags = self.create_header_data(headers)
        self.names = get_unique_names_duplicates(names)
        self.types = types
        self.flags = flags

    @classmethod
    def create_header_data(cls, headers: List) -> Tuple[List, List, List]:
        """
        Consider various header types (single-row, two-row, three-row, none).

        Parameters
        ----------
        headers: List
            Header rows, to be used for constructing domain.

        Returns
        -------
        names: List
            List of variable names.
        types: List
            List of variable types.
        flags: List
            List of meta info (i.e. class, meta, ignore, weights).
        """
        return {3: lambda x: x,
                2: cls._header2,
                1: cls._header1}.get(len(headers), cls._header0)(headers)

    @classmethod
    def _header2(cls, headers: List[List[str]]) -> Tuple[List, List, List]:
        names, flags = headers
        return names, cls._type_from_flag(flags), cls._flag_from_flag(flags)

    @classmethod
    def _header1(cls, headers: List[List[str]]) -> Tuple[List, List, List]:
        """
        First row format either:
          1) delimited column names
          2) -||- with type and flags prepended, separated by #,
             e.g. d#sex,c#age,cC#IQ
        """
        flags, names = zip(*[i.split(cls.HEADER1_FLAG_SEP, 1)
                             if cls.HEADER1_FLAG_SEP in i else ('', i)
                             for i in headers[0]])
        names = list(names)
        return names, cls._type_from_flag(flags), cls._flag_from_flag(flags)

    @classmethod
    def _header0(cls, _) -> Tuple[List, List, List]:
        # Use heuristics for everything
        return [], [], []

    @staticmethod
    def _type_from_flag(flags: List[str]) -> List[str]:
        return [''.join(filter(str.isupper, flag)).lower() for flag in flags]

    @staticmethod
    def _flag_from_flag(flags: List[str]) -> List[str]:
        return [Flags.join(filter(str.islower, flag)) for flag in flags]


class _TableBuilder:
    X_ARR, Y_ARR, M_ARR, W_ARR = range(4)
    DATA_IND, DOMAIN_IND, TYPE_IND = range(3)

    def __init__(self, data: np.ndarray, ncols: int,
                 header: _TableHeader, offset: int):
        self.data = data
        self.ncols = ncols
        self.header = header
        self.offset = offset
        self.namegen: Generator[str] = namegen('Feature ', 1)

        self.cols_X: List[np.ndarray] = []
        self.cols_Y: List[np.ndarray] = []
        self.cols_M: List[np.ndarray] = []
        self.cols_W: List[np.ndarray] = []
        self.attrs: List[Variable] = []
        self.clses: List[Variable] = []
        self.metas: List[Variable] = []

    def create_table(self) -> Table:
        self.create_columns()
        if not self.data.size:
            return Table.from_domain(self.get_domain(), 0)
        else:
            return Table.from_numpy(self.get_domain(), *self.get_arrays())

    def create_columns(self):
        names = self.header.names
        types = self.header.types

        for col in range(self.ncols):
            flag = Flags(Flags.split(self.header.flags[col]))
            if flag.i:
                continue

            type_ = types and types[col].strip()
            creator = self._get_column_creator(type_)
            column = creator(self.data, col, values=type_, offset=self.offset)
            self._take_column(names and names[col], column, flag)
            self._reclaim_memory(self.data, col)

    @classmethod
    def _get_column_creator(cls, type_: str) -> Callable:
        if type_ in StringVariable.TYPE_HEADERS:
            return cls._string_column
        elif type_ in ContinuousVariable.TYPE_HEADERS:
            return cls._cont_column
        elif type_ in TimeVariable.TYPE_HEADERS:
            return cls._time_column
        elif _RE_DISCRETE_LIST.match(type_):
            return cls._disc_with_vals_column
        elif type_ in DiscreteVariable.TYPE_HEADERS:
            return cls._disc_no_vals_column
        else:
            return cls._unknown_column

    @staticmethod
    def _string_column(data: np.ndarray, col: int, **_) -> _ColumnProperties:
        vals, _ = _TableBuilder._values_mask(data, col)
        return _ColumnProperties(values=vals, coltype=StringVariable,
                                 orig_values=vals)

    @staticmethod
    def _cont_column(data: np.ndarray, col: int,
                     offset=0, **_) -> _ColumnProperties:
        orig_vals, namask = _TableBuilder._values_mask(data, col)
        values = np.empty(data.shape[0], dtype=float)
        try:
            np.copyto(values, orig_vals, casting="unsafe", where=~namask)
            values[namask] = np.nan
        except ValueError:
            row = 0
            for row, num in enumerate(orig_vals):
                if not isnastr(num):
                    try:
                        float(num)
                    except ValueError:
                        break
            raise ValueError(f'Non-continuous value in (1-based) '
                             f'line {row + offset + 1}, column {col + 1}')
        return _ColumnProperties(values=values, coltype=ContinuousVariable,
                                 orig_values=orig_vals)

    @staticmethod
    def _time_column(data: np.ndarray, col: int, **_) -> _ColumnProperties:
        vals, namask = _TableBuilder._values_mask(data, col)
        return _ColumnProperties(values=np.where(namask, "", vals),
                                 coltype=TimeVariable, orig_values=vals)

    @staticmethod
    def _disc_column(data: np.ndarray, col: int) -> \
            Tuple[np.ndarray, VariableMeta]:
        vals, namask = _TableBuilder._values_mask(data, col)
        return np.where(namask, "", vals), DiscreteVariable

    @staticmethod
    def _disc_no_vals_column(data: np.ndarray, col: int, **_) -> \
            _ColumnProperties:
        vals, coltype = _TableBuilder._disc_column(data, col)
        return _ColumnProperties(valuemap=natural_sorted(set(vals) - {""}),
                                 values=vals, coltype=coltype,
                                 orig_values=vals)

    @staticmethod
    def _disc_with_vals_column(data: np.ndarray, col: int,
                               values="", **_) -> _ColumnProperties:
        vals, coltype = _TableBuilder._disc_column(data, col)
        return _ColumnProperties(valuemap=Flags.split(values), values=vals,
                                 coltype=coltype, orig_values=vals)

    @staticmethod
    def _unknown_column(data: np.ndarray, col: int, **_) -> _ColumnProperties:
        orig_vals, namask = _TableBuilder._values_mask(data, col)
        valuemap, values, coltype = guess_data_type(orig_vals, namask)
        return _ColumnProperties(valuemap=valuemap, values=values,
                                 coltype=coltype, orig_values=orig_vals)

    @staticmethod
    def _values_mask(data: np.ndarray, col: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        try:
            values = data[:, col]
        except IndexError:
            values = np.array([], dtype=object)
        return values, isnastr(values)

    def _take_column(self, name: Optional[str], column: _ColumnProperties,
                     flag: Flags):
        cols, dom_vars = self._lists_from_flag(flag, column.coltype)
        values = column.values
        if dom_vars is not None:
            if not name:
                name = next(self.namegen)

            values, var = sanitize_variable(
                column.valuemap, values, column.orig_values,
                column.coltype, column.coltype_kwargs, name=name)
            var.attributes.update(flag.attributes)
            dom_vars.append(var)

        if isinstance(values, np.ndarray) and not values.flags.owndata:
            values = values.copy()  # might view `data` (string columns)
        cols.append(values)

    def _lists_from_flag(self, flag: Flags, coltype: VariableMeta) -> \
            Tuple[List, Optional[List]]:
        if flag.m or coltype is StringVariable:
            return self.cols_M, self.metas
        elif flag.w:
            return self.cols_W, None
        elif flag.c:
            return self.cols_Y, self.clses
        else:
            return self.cols_X, self.attrs

    @staticmethod
    def _reclaim_memory(data: np.ndarray, col: int):
        # allow gc to reclaim memory used by string values
        try:
            data[:, col] = None
        except IndexError:
            pass

    def get_domain(self) -> Domain:
        return Domain(self.attrs, self.clses, self.metas)

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
        lists = ((self.cols_X, None),
                 (self.cols_Y, None),
                 (self.cols_M, object),
                 (self.cols_W, float))
        X, Y, M, W = [self._list_into_ndarray(lst, dt) for lst, dt in lists]
        if X is None:
            X = np.empty((self.data.shape[0], 0), dtype=np.float_)
        return X, Y, M, W

    @staticmethod
    def _list_into_ndarray(lst: List, dtype=None) -> Optional[np.ndarray]:
        if not lst:
            return None

        array = np.c_[tuple(lst)]
        if dtype is not None:
            array.astype(dtype)
        else:
            assert array.dtype == np.float_
        return array


class DataTableMixin:
    @classmethod
    def data_table(cls, data: Iterable[List[str]],
                   headers: Optional[List] = None) -> Table:
        """
        Return Orange.data.Table given rows of `headers` (iterable of iterable)
        and rows of `data` (iterable of iterable).

        Basically, the idea of subclasses is to produce those two iterables,
        however they might.

        If `headers` is not provided, the header rows are extracted from `data`,
        assuming they precede it.

        Parameters
        ----------
        data: Iterable
            File content.
        headers: List (Optional)
            Header rows, to be used for constructing domain.

        Returns
        -------
        table: Table
            Data as Orange.data.Table.
        """
        if not headers:
            headers, data = cls.parse_headers(data)

        header = _TableHeader(headers)
        # adjusting data may change header properties
        array, n_columns = cls.adjust_data_width(data, header)
        builder = _TableBuilder(array, n_columns, header, len(headers))
        return builder.create_table()

    @classmethod
    def parse_headers(cls, data: Iterable[List[str]]) -> Tuple[List, Iterable]:
        """
        Return (header rows, rest of data) as discerned from `data`.

        Parameters
        ----------
        data: Iterable
            File content.

        Returns
        -------
        header_rows: List
            Header rows, to be used for constructing domain.
        data: Iterable
            File content without header rows.
        """
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
            if cls.__header_test2(l2) and cls.__header_test3(l3):
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
                if not all(cls.__is_number(i) for i in lines[0]):
                    header_rows = [lines[0]]
                else:
                    data = chain(lines, data)

        return header_rows, data

    @staticmethod
    def __is_number(item: str) -> bool:
        try:
            float(item)
        except ValueError:
            return False
        return True

    @staticmethod
    def __header_test2(items: List) -> bool:
        # Second row items are type identifiers
        return all(map(_RE_TYPES.match, items))

    @staticmethod
    def __header_test3(items: List) -> bool:
        # Third row items are flags and column attributes (attr=value)
        return all(map(_RE_FLAGS.match, items))

    @classmethod
    def adjust_data_width(cls, data: Iterable, header: _TableHeader) -> \
            Tuple[np.ndarray, int]:
        """
        Determine maximum row length.
        Return data as an array, with width dependent on header size.
        Append `names`, `types` and `flags` if shorter than row length.

        Parameters
        ----------
        data: Iterable
            File content without header rows.
        header: _TableHeader
            Header lists converted into _TableHeader.

        Returns
        -------
        data: np.ndarray
            File content without header rows.
        rowlen: int
            Number of columns in data.
        """

        def equal_len(lst):
            nonlocal strip
            if len(lst) > rowlen > 0:
                lst = lst[:rowlen]
                strip = True
            elif len(lst) < rowlen:
                lst.extend([''] * (rowlen - len(lst)))
            return lst

        rowlen = max(map(len, (header.names, header.types, header.flags)))
        strip = False

        # Ensure all data is of equal width in a column-contiguous array
        data = [equal_len([s.strip() for s in row])
                for row in data if any(row)]
        array = np.array(data, dtype=object, order='F')

        if strip:
            warnings.warn("Columns with no headers were removed.")

        # Data may actually be longer than headers were
        try:
            rowlen = array.shape[1]
        except IndexError:
            pass
        else:
            for lst in (header.names, header.types, header.flags):
                equal_len(lst)
        return array, rowlen


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
                    pickle.dump(data.attributes, f, protocol=PICKLE_PROTOCOL)

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
                # if number of values is 1 order is not important if more
                # values write order in file
                return (
                    Flags.join(var.values) if len(var.values) >= 2
                    else var.TYPE_HEADERS[0]
                )
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
