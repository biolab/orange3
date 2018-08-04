import contextlib
import csv
import xlrd
import xlwt
import locale
import pickle
import re
import subprocess
import sys
import warnings

from ast import literal_eval
from collections import OrderedDict, Counter, defaultdict
from functools import lru_cache
from importlib import import_module
from itertools import chain, repeat
from math import isnan
from pyexcel_ods import save_data as ods_write

from os import path, remove
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse, urlsplit, urlunsplit, unquote as urlunquote
from urllib.request import urlopen, Request
from fnmatch import fnmatch
from glob import glob

import numpy as np
from chardet.universaldetector import UniversalDetector

from Orange.data import (
    _io, is_discrete_values, MISSING_VALUES, Table, Domain, Variable,
    DiscreteVariable, StringVariable, ContinuousVariable, TimeVariable,
)
from Orange.util import Registry, flatten, namegen


# Support values longer than 128K (i.e. text contents features)
csv.field_size_limit(100*1024*1024)

__all__ = ["Flags", "FileFormat"]


_IDENTITY = lambda i: i


class Compression:
    """Supported compression extensions"""
    GZIP = '.gz'
    BZIP2 = '.bz2'
    XZ = '.xz'
    all = (GZIP, BZIP2, XZ)


def open_compressed(filename, *args, _open=open, **kwargs):
    """Return seamlessly decompressed open file handle for `filename`"""
    if isinstance(filename, str):
        if filename.endswith(Compression.GZIP):
            from gzip import open as _open
        elif filename.endswith(Compression.BZIP2):
            from bz2 import open as _open
        elif filename.endswith(Compression.XZ):
            from lzma import open as _open
        return _open(filename, *args, **kwargs)
    # Else already a file, just pass it through
    return filename


def detect_encoding(filename):
    """
    Detect encoding of `filename`, which can be a ``str`` filename, a
    ``file``-like object, or ``bytes``.
    """
    # Try with Unix file utility first because it's faster (~10ms vs 100ms)
    if isinstance(filename, str) and not filename.endswith(Compression.all):
        try:
            with subprocess.Popen(('file', '--brief', '--mime-encoding', filename),
                                  stdout=subprocess.PIPE) as process:
                process.wait()
                if process.returncode == 0:
                    encoding = process.stdout.read().strip()
                    # file only supports these encodings; for others it says
                    # unknown-8bit or binary. So we give chardet a chance to do
                    # better
                    if encoding in (b'utf-8', b'us-ascii', b'iso-8859-1',
                                    b'utf-7', b'utf-16le', b'utf-16be', b'ebcdic'):
                        return encoding.decode('us-ascii')
        except OSError:
            pass  # windoze

    # file not available or unable to guess the encoding, have chardet do it
    detector = UniversalDetector()
    # We examine only first N 4kB blocks of file because chardet is really slow
    MAX_BYTES = 4*1024*12

    def _from_file(f):
        detector.feed(f.read(MAX_BYTES))
        detector.close()
        return (detector.result.get('encoding')
                if detector.result.get('confidence', 0) >= .85 else
                'utf-8')

    if isinstance(filename, str):
        with open_compressed(filename, 'rb') as f:
            return _from_file(f)
    elif isinstance(filename, bytes):
        detector.feed(filename[:MAX_BYTES])
        detector.close()
        return detector.result.get('encoding')
    elif hasattr(filename, 'encoding'):
        return filename.encoding
    else:  # assume file-like object that you can iter through
        return _from_file(filename)


__isnastr = np.frompyfunc(
    {v for v in MISSING_VALUES if isinstance(v, str)}.__contains__, 1, 1)

# wrapper for __isnastr with proper default out dtype
def isnastr(arr, out=None):
    """
    Given an (object) array of string values, return a boolean mask array
    that is True where the `arr` contains one of the string constants
    considered as N/A.

    Parameters
    ----------
    arr : np.ndarray
        Input array of strings.
    out : Optional[np.ndarray]
        Optional output array of the same shape as arr

    Returns
    -------
    mask : np.ndarray
    """
    arr = np.asarray(arr)
    if out is None and arr.shape != ():
        out = np.empty_like(arr, dtype=bool)
    return __isnastr(arr, out=out)


def guess_data_type(orig_values, namask=None):
    """
    Use heuristics to guess data type.
    """
    valuemap, values = None, orig_values
    is_discrete = is_discrete_values(orig_values)
    if is_discrete:
        valuemap = sorted(is_discrete)
        coltype = DiscreteVariable
    else:
        # try to parse as float
        orig_values = np.asarray(orig_values)
        if namask is None:
            namask = isnastr(orig_values)
        values = np.empty_like(orig_values, dtype=float)
        values[namask] = np.nan
        try:
            np.copyto(values, orig_values, where=~namask, casting="unsafe")
        except ValueError:
            tvar = TimeVariable('_')
            try:
                values[~namask] = [tvar.parse(i) for i in orig_values[~namask]]
            except ValueError:
                coltype = StringVariable
                # return original_values
                values = orig_values
            else:
                coltype = TimeVariable
        else:
            coltype = ContinuousVariable
    return valuemap, values, coltype


def sanitize_variable(valuemap, values, orig_values, coltype, coltype_kwargs,
                      domain_vars, existing_var, new_var_name, data=None):
    assert issubclass(coltype, Variable)

    def get_number_of_decimals(values):
        len_ = len
        ndecimals = max((len_(value) - value.find(".")
                         for value in values if "." in value),
                        default=1)
        return ndecimals - 1

    if issubclass(coltype, DiscreteVariable) and valuemap is not None:
        coltype_kwargs.update(values=valuemap)

    if existing_var:
        # Use existing variable if available
        var = coltype.make(existing_var.strip(), **coltype_kwargs)
    else:
        # Never use existing for un-named variables
        var = coltype(new_var_name, **coltype_kwargs)

    if isinstance(var, DiscreteVariable):
        # Map discrete data to 'ints' (or at least what passes as int around
        # here)
        mapping = defaultdict(
            lambda: np.nan,
            {val: i for i, val in enumerate(var.values)},
        )
        mapping[""] = np.nan
        mapvalues_ = np.frompyfunc(mapping.__getitem__, 1, 1)

        def mapvalues(arr):
            arr = np.asarray(arr, dtype=object)
            return mapvalues_(arr, out=np.empty_like(arr, dtype=float))
        values = mapvalues(orig_values)

    if coltype is StringVariable:
        values = orig_values

    # ContinuousVariable.number_of_decimals is supposed to be handled by
    # ContinuousVariable.to_val. In the interest of speed, the reader bypasses
    # it, so we set the number of decimals here.
    # The number of decimals is increased if not set manually (in which case
    # var.adjust_decimals would be 0).
    if isinstance(var, ContinuousVariable) and var.adjust_decimals:
        ndecimals = get_number_of_decimals(orig_values)
        if var.adjust_decimals == 2 or ndecimals > var.number_of_decimals:
            var.number_of_decimals = ndecimals
            var.adjust_decimals = 1

    if isinstance(var, TimeVariable) or coltype is TimeVariable:
        # Re-parse the values because only now after coltype.make call
        # above, variable var is the correct one
        _var = var if isinstance(var, TimeVariable) else TimeVariable('_')
        values = [_var.parse(i) for i in orig_values]

    return values, var


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
    _RE_ALL = re.compile(r'^({})$'.format('|'.join(filter(None, flatten(ALL.items())))))

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
        return Flags.DELIMITER.join(i.strip().replace(Flags.DELIMITER, '\\' + Flags.DELIMITER)
                                    for i in chain(iterable, args)).lstrip()

    @staticmethod
    def split(s):
        return [i.replace('\\' + Flags.DELIMITER, Flags.DELIMITER)
                for i in Flags._RE_SPLIT(s)]


# Matches discrete specification where all the values are listed, space-separated
_RE_DISCRETE_LIST = re.compile(r'^\s*[^\s]+(\s[^\s]+)+\s*$')
_RE_TYPES = re.compile(r'^\s*({}|{}|)\s*$'.format(
    _RE_DISCRETE_LIST.pattern,
    '|'.join(flatten(getattr(vartype, 'TYPE_HEADERS') for vartype in Variable.registry.values()))
    ))
_RE_FLAGS = re.compile(r'^\s*( |{}|)*\s*$'.format(
    '|'.join(flatten(filter(None, i) for i in Flags.ALL.items()))
    ))


class FileFormatMeta(Registry):

    def __new__(cls, name, bases, attrs):
        newcls = super().__new__(cls, name, bases, attrs)

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
        for format in sorted(cls.registry.values(), key=lambda x: x.PRIORITY):
            if not hasattr(format, attr2):
                continue
            for ext in getattr(format, 'EXTENSIONS', []):
                # Only adds if not yet registered
                formats.setdefault(ext, getattr(format, attr, format))
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
        return cls._ext_to_attr_if_attr2('', 'write_image')

    @property
    def graph_writers(cls):
        return cls._ext_to_attr_if_attr2('', 'write_graph')


class FileFormat(metaclass=FileFormatMeta):
    """
    Subclasses set the following attributes and override the following methods:

        EXTENSIONS = ('.ext1', '.ext2', ...)
        DESCRIPTION = 'human-readable file format description'
        SUPPORT_COMPRESSED = False
        SUPPORT_SPARSE_DATA = False

        def read(self):
            ...  # load headers, data, ...
            return self.data_table(data, headers)

        @classmethod
        def write_file(cls, filename, data):
            ...
            self.write_headers(writer.write, data)
            writer.writerows(data)

    Wrapper FileFormat.data_table() returns Orange.data.Table from `data`
    iterable (list (rows) of lists of values (cols)).
    """

    # Priority when multiple formats support the same extension. Also
    # the sort order in file open/save combo boxes. Lower is better.
    PRIORITY = 10000

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            name of the file to open
        """
        self.filename = filename
        self.sheet = None

    @property
    def sheets(self):
        """FileFormats with a notion of sheets should override this property
        to return a list of sheet names in the file.

        Returns
        -------
        a list of sheet names
        """
        return ()

    def select_sheet(self, sheet):
        """Select sheet to be read

        Parameters
        ----------
        sheet : str
            sheet name
        """
        self.sheet = sheet

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
    def write(cls, filename, data):
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

        If `headers` is not provided, the header rows are extracted from `data`,
        assuming they precede it.
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
            types = [''.join(filter(str.isupper, flag)).lower() for flag in _flags]
            flags = [Flags.join(filter(str.islower, flag)) for flag in _flags]

        # Determine maximum row length
        rowlen = max(map(len, (names, types, flags)))

        def _equal_length(lst):
            lst.extend(['']*(rowlen - len(lst)))
            return lst

        # Ensure all data is of equal width in a column-contiguous array
        data = [_equal_length([s.strip() for s in row])
                for row in data if any(row)]
        data = np.array(data, dtype=object, order='F')

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
            uses = {name: 0 for name, count in name_counts.items() if count > 1}
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
                    raise ValueError('Non-continuous value in (1-based) '
                                     'line {}, column {}'.format(row + len(headers) + 1,
                                                                 col + 1))

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
                valuemap, values, coltype = guess_data_type(orig_values, namask)

            if flag.m or coltype is StringVariable:
                append_to = (Mcols, metas)
            elif flag.w:
                append_to = (Wcols, None)
            elif flag.c:
                append_to = (Ycols, clses)
            else:
                append_to = (Xcols, attrs)

            cols, domain_vars = append_to

            existing_var, new_var_name = None, None
            if domain_vars is not None:
                existing_var = names and names[col]
                if not existing_var:
                    new_var_name = next(NAMEGEN)

            if domain_vars is not None:
                values, var = sanitize_variable(
                    valuemap, values, orig_values, coltype, coltype_kwargs,
                    domain_vars, existing_var, new_var_name, data)
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
                return Flags.join(var.values) if var.ordered else var.TYPE_HEADERS[0]
            raise NotImplementedError
        return ['continuous'] * data.has_weights() + \
               [_vartype(v) for v in chain(data.domain.attributes,
                                           data.domain.class_vars,
                                           data.domain.metas)]

    @staticmethod
    def header_flags(data):
        return list(chain(['weight'] * data.has_weights(),
                          (Flags.join([flag], *('{}={}'.format(*a)
                                                for a in sorted(var.attributes.items())))
                           for flag, var in chain(zip(repeat(''), data.domain.attributes),
                                                  zip(repeat('class'), data.domain.class_vars),
                                                  zip(repeat('meta'), data.domain.metas)))))

    @classmethod
    def write_headers(cls, write, data):
        """`write` is a callback that accepts an iterable"""
        write(cls.header_names(data))
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
        vars = list(chain((ContinuousVariable('_w'),) if data.has_weights() else (),
                          data.domain.attributes,
                          data.domain.class_vars,
                          data.domain.metas))

        formatters = [cls.formatter(v) for v in vars]
        for row in zip(data.W if data.W.ndim > 1 else data.W[:, np.newaxis],
                       data.X,
                       data.Y if data.Y.ndim > 1 else data.Y[:, np.newaxis],
                       data.metas):
            write([fmt(v) for fmt, v in zip(formatters, flatten(row))])

    @classmethod
    def qualified_name(cls):
        return cls.__module__ + '.' + cls.__name__


def class_from_qualified_name(format_name):
    """ File format class from qualified name. """
    elements = format_name.split(".")
    m = import_module(".".join(elements[:-1]))
    return getattr(m, elements[-1])


class CSVReader(FileFormat):
    """Reader for comma separated files"""

    EXTENSIONS = ('.csv',)
    DESCRIPTION = 'Comma-separated values'
    DELIMITERS = ',;:\t$ '
    SUPPORT_COMPRESSED = True
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 20

    def read(self):
        for encoding in (lambda: ('us-ascii', None),                 # fast
                         lambda: (detect_encoding(self.filename), None),  # precise
                         lambda: (locale.getpreferredencoding(False), None),
                         lambda: (sys.getdefaultencoding(), None),   # desperate
                         lambda: ('utf-8', None),                    # ...
                         lambda: ('utf-8', 'ignore')):               # fallback
            encoding, errors = encoding()
            # Clear the error flag for all except the last check, because
            # the error of second-to-last check is stored and shown as warning in owfile
            if errors != 'ignore':
                error = ''
            with self.open(self.filename, mode='rt', newline='',
                           encoding=encoding, errors=errors) as file:
                # Sniff the CSV dialect (delimiter, quotes, ...)
                try:
                    dialect = csv.Sniffer().sniff(
                        # Take first couple of *complete* lines as sample
                        ''.join(file.readline() for _ in range(10)),
                        self.DELIMITERS)
                    delimiter = dialect.delimiter
                    quotechar = dialect.quotechar
                except UnicodeDecodeError as e:
                    error = e
                    continue
                except csv.Error:
                    delimiter = self.DELIMITERS[0]
                    quotechar = csv.excel.quotechar

                file.seek(0)
                try:
                    reader = csv.reader(
                        file, delimiter=delimiter, quotechar=quotechar,
                        skipinitialspace=True,
                    )
                    data = self.data_table(reader)

                    # TODO: Name can be set unconditionally when/if
                    # self.filename will always be a string with the file name.
                    # Currently, some tests pass StringIO instead of
                    # the file name to a reader.
                    if isinstance(self.filename, str):
                        data.name = path.splitext(
                            path.split(self.filename)[-1])[0]
                    if error and isinstance(error, UnicodeDecodeError):
                        pos, endpos = error.args[2], error.args[3]
                        warning = ('Skipped invalid byte(s) in position '
                                   '{}{}').format(pos,
                                                  ('-' + str(endpos)) if (endpos - pos) > 1 else '')
                        warnings.warn(warning)
                    self.set_table_metadata(self.filename, data)
                    return data
                except Exception as e:
                    error = e
                    continue
        raise ValueError('Cannot parse dataset {}: {}'.format(self.filename, error)) from error

    @classmethod
    def write_file(cls, filename, data):
        with cls.open(filename, mode='wt', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=cls.DELIMITERS[0])
            cls.write_headers(writer.writerow, data)
            cls.write_data(writer.writerow, data)
        cls.write_table_metadata(filename, data)


class TabReader(CSVReader):
    """Reader for tab separated files"""
    EXTENSIONS = ('.tab', '.tsv')
    DESCRIPTION = 'Tab-separated values'
    DELIMITERS = '\t'
    PRIORITY = 10


class PickleReader(FileFormat):
    """Reader for pickled Table objects"""
    EXTENSIONS = ('.pkl', '.pickle')
    DESCRIPTION = 'Pickled Python object file'
    SUPPORT_COMPRESSED = True
    SUPPORT_SPARSE_DATA = True

    def read(self):
        with self.open(self.filename, 'rb') as f:
            table = pickle.load(f)
            if not isinstance(table, Table):
                raise TypeError("file does not contain a data table")
            else:
                return table

    @classmethod
    def write_file(cls, filename, data):
        with cls.open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class BasketReader(FileFormat):
    """Reader for basket (sparse) files"""
    EXTENSIONS = ('.basket', '.bsk')
    DESCRIPTION = 'Basket file'
    SUPPORT_SPARSE_DATA = True

    def read(self):
        def constr_vars(inds):
            if inds:
                return [ContinuousVariable(x.decode("utf-8")) for _, x in
                        sorted((ind, name) for name, ind in inds.items())]

        X, Y, metas, attr_indices, class_indices, meta_indices = \
            _io.sparse_read_float(self.filename.encode(sys.getdefaultencoding()))

        attrs = constr_vars(attr_indices)
        classes = constr_vars(class_indices)
        meta_attrs = constr_vars(meta_indices)
        domain = Domain(attrs, classes, meta_attrs)
        table = Table.from_numpy(
            domain, attrs and X, classes and Y, metas and meta_attrs)
        table.name = path.splitext(path.split(self.filename)[-1])[0]
        return table


class ExcelReader(FileFormat):
    """Reader for excel files"""
    EXTENSIONS = ('.xls', '.xlsx', '.ods')
    DESCRIPTION = 'Mircosoft Excel spreadsheet'
    SUPPORT_SPARSE_DATA = False

    def __init__(self, filename):
        super().__init__(filename)
        self.workbook = xlrd.open_workbook(self.filename)

    @property
    @lru_cache(1)
    def sheets(self):
        return self.workbook.sheet_names()

    def read(self):
        wb = xlrd.open_workbook(self.filename, on_demand=True)
        if self.sheet:
            ss = wb.sheet_by_name(self.sheet)
        else:
            ss = wb.sheet_by_index(0)
        try:
            first_row = next(i for i in range(ss.nrows) if any(ss.row_values(i)))
            first_col = next(i for i in range(ss.ncols) if ss.cell_value(first_row, i))
            row_len = ss.row_len(first_row)
            cells = filter(any,
                           [[str(ss.cell_value(row, col)) if col < ss.row_len(row) else ''
                             for col in range(first_col, row_len)]
                            for row in range(first_row, ss.nrows)])
            table = self.data_table(cells)
            table.name = path.splitext(path.split(self.filename)[-1])[0]
            if self.sheet:
                table.name = '-'.join((table.name, self.sheet))
        except Exception:
            raise IOError("Couldn't load spreadsheet from " + self.filename)
        return table

    @classmethod
    def write_file(cls, filename, data):
        vars = list(chain((ContinuousVariable('_w'),) if data.has_weights() else (),
                          data.domain.attributes,
                          data.domain.class_vars,
                          data.domain.metas))
        formatters = [cls.formatter(v) for v in vars]
        zipped_list_data = zip(data.W if data.W.ndim > 1 else data.W[:, np.newaxis],
                       data.X,
                       data.Y if data.Y.ndim > 1 else data.Y[:, np.newaxis],
                       data.metas)
        headers = cls.header_names(data)
        if filename.endswith((".xls", ".xlsx")):
            workbook = xlwt.Workbook(encoding = "utf-8") 
            sheet = workbook.add_sheet("Sheet1", cell_overwrite_ok=True)
            for c, header in enumerate(headers): 
                sheet.write(0, c, header)
            for i, row in enumerate(zipped_list_data):
                j = 0
                for fmt, v in zip(formatters, flatten(row)):
                    sheet.write(i+1, j, fmt(v))
                    j += 1
            workbook.save(filename)
        elif filename.endswith(".ods"):
            ods_formatted_data = [[str(fmt(v)) for fmt, v in zip(formatters, flatten(row))] for row in zipped_list_data]
            ods_write(filename, {"Sheet1": [headers] + ods_formatted_data})


class DotReader(FileFormat):
    """Writer for dot (graph) files"""
    EXTENSIONS = ('.dot', '.gv')
    DESCRIPTION = 'Dot graph description'
    SUPPORT_COMPRESSED = True
    SUPPORT_SPARSE_DATA = False

    @classmethod
    def write_graph(cls, filename, graph):
        from sklearn import tree
        tree.export_graphviz(graph, out_file=cls.open(filename, 'wt'))

    @classmethod
    def write(cls, filename, tree):
        if type(tree) == dict:
            tree = tree['tree']
        cls.write_graph(filename, tree)


class UrlReader(FileFormat):
    def __init__(self, filename):
        filename = filename.strip()
        if not urlparse(filename).scheme:
            filename = 'http://' + filename
        super().__init__(filename)

    @staticmethod
    def urlopen(url):
        req = Request(
            url,
            # Avoid 403 error with servers that dislike scrapers
            headers={'User-Agent': 'Mozilla/5.0 (X11; Linux) Gecko/20100101 Firefox/'})
        return urlopen(req, timeout=10)

    def read(self):
        self.filename = self._trim(self._resolve_redirects(self.filename))
        with contextlib.closing(self.urlopen(self.filename)) as response:
            name = self._suggest_filename(response.headers['content-disposition'])
            with NamedTemporaryFile(suffix=name, delete=False) as f:
                f.write(response.read())
                # delete=False is a workaround for https://bugs.python.org/issue14243

            reader = self.get_reader(f.name)
            data = reader.read()
            remove(f.name)
        # Override name set in from_file() to avoid holding the temp prefix
        data.name = path.splitext(name)[0]
        data.origin = self.filename
        return data

    def _resolve_redirects(self, url):
        # Resolve (potential) redirects to a final URL
        with contextlib.closing(self.urlopen(url)) as response:
            return response.url

    @classmethod
    def _trim(cls, url):
        URL_TRIMMERS = (
            cls._trim_googlesheet,
            cls._trim_dropbox,
        )
        for trim in URL_TRIMMERS:
            try:
                url = trim(url)
            except ValueError:
                continue
            else:
                break
        return url

    @staticmethod
    def _trim_googlesheet(url):
        match = re.match(r'(?:https?://)?(?:www\.)?'
                         r'docs\.google\.com/spreadsheets/d/'
                         r'(?P<workbook_id>[-\w_]+)'
                         r'(?:/.*?gid=(?P<sheet_id>\d+).*|.*)?',
                         url, re.IGNORECASE)
        try:
            workbook, sheet = match.group('workbook_id'), match.group('sheet_id')
            if not workbook:
                raise ValueError
        except (AttributeError, ValueError):
            raise ValueError
        url = 'https://docs.google.com/spreadsheets/d/{}/export?format=tsv'.format(workbook)
        if sheet:
            url += '&gid=' + sheet
        return url

    @staticmethod
    def _trim_dropbox(url):
        parts = urlsplit(url)
        if not parts.netloc.endswith('dropbox.com'):
            raise ValueError
        return urlunsplit(parts._replace(query='dl=1'))

    def _suggest_filename(self, content_disposition):
        default_name = re.sub(r'[\\:/]', '_', urlparse(self.filename).path)

        # See https://tools.ietf.org/html/rfc6266#section-4.1
        matches = re.findall(r"filename\*?=(?:\"|.{0,10}?'[^']*')([^\"]+)",
                             content_disposition or '')
        return urlunquote(matches[-1]) if matches else default_name
