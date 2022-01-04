import subprocess
from collections import defaultdict

import numpy as np
from chardet.universaldetector import UniversalDetector

from Orange.data import (
    is_discrete_values, MISSING_VALUES, Variable,
    DiscreteVariable, StringVariable, ContinuousVariable, TimeVariable,
)
from Orange.misc.collections import natural_sorted

__all__ = ["Compression", "open_compressed", "detect_encoding", "isnastr",
           "guess_data_type", "sanitize_variable"]


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
            with subprocess.Popen(('file', '--brief', '--mime-encoding',
                                   filename), stdout=subprocess.PIPE) as proc:
                proc.wait()
                if proc.returncode == 0:
                    encoding = proc.stdout.read().strip()
                    # file only supports these encodings; for others it says
                    # unknown-8bit or binary. So we give chardet a chance to do
                    # better
                    if encoding in (b'utf-8', b'us-ascii', b'iso-8859-1',
                                    b'utf-7', b'utf-16le', b'utf-16be',
                                    b'ebcdic'):
                        return encoding.decode('us-ascii')
        except OSError:
            pass  # windoze

    # file not available or unable to guess the encoding, have chardet do it
    detector = UniversalDetector()
    # We examine only first N 4kB blocks of file because chardet is really slow
    MAX_BYTES = 4 * 1024 * 12

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
    return __isnastr(arr, out=out, casting="unsafe")


def guess_data_type(orig_values, namask=None):
    """
    Use heuristics to guess data type.
    """
    valuemap, values = None, orig_values
    is_discrete = is_discrete_values(orig_values)
    orig_values = np.asarray(orig_values, dtype=str)
    if namask is None:
        namask = isnastr(orig_values)
    if is_discrete:
        valuemap = natural_sorted(is_discrete)
        coltype = DiscreteVariable
    else:
        # try to parse as float
        values = np.empty_like(orig_values, dtype=float)
        values[namask] = np.nan
        try:
            np.copyto(values, orig_values, where=~namask, casting="unsafe")
        except ValueError:
            values = orig_values
            coltype = StringVariable
        else:
            coltype = ContinuousVariable

    if coltype is not ContinuousVariable:
        # when not continuous variable it can still be time variable even it
        # was before recognized as a discrete
        tvar = TimeVariable('_')
        # introducing new variable prevent overwriting orig_values and values
        temp_values = np.empty_like(orig_values, dtype=float)
        try:
            temp_values[~namask] = [
                tvar.parse_exact_iso(i) for i in orig_values[~namask]]
        except ValueError:
            pass
        else:
            valuemap = None
            coltype = TimeVariable
            values = temp_values
    return valuemap, values, coltype


def sanitize_variable(valuemap, values, orig_values, coltype, coltype_kwargs,
                      name=None):
    assert issubclass(coltype, Variable)

    def get_number_of_decimals(values):
        len_ = len
        ndecimals = max((len_(value) - value.find(".")
                         for value in values if "." in value),
                        default=1)
        return ndecimals - 1

    if issubclass(coltype, DiscreteVariable) and valuemap is not None:
        coltype_kwargs.update(values=valuemap)

    var = coltype.make(name, **coltype_kwargs)

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
            return mapvalues_(arr, out=np.empty_like(arr, dtype=float), casting="unsafe")

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
