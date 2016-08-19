import unittest
from functools import wraps
from time import perf_counter
import platform

import numpy as np

from Orange.data import Table

# override method prefix for niceness
BENCH_METHOD_PREFIX = 'bench'
unittest.TestLoader.testMethodPrefix = BENCH_METHOD_PREFIX


def _timeitlike_time_format(time_seconds, precision=3):
    """Shamelessly adapted formatting from timeit.py

    Parameters
    ----------
    time_seconds : float
        The time in seconds
    precision : int
        The precision of the output. All digits.

    Returns
    -------
    str
        A timeit-like format (with usec, msec, sec).
    """
    usec = time_seconds * 1e6
    if usec < 1000:
        return "%.*g usec" % (precision, usec)
    else:
        msec = usec / 1000
        if msec < 1000:
            return "%.*g msec" % (precision, msec)
        else:
            sec = msec / 1000
            return "%.*g sec" % (precision, sec)


def _bench_skipper(condition, skip_message):
    """A decorator factory for sipping benchmarks conditionally.

    Parameters
    ----------
    condition : bool or function
        A boolean value or a lambda callback to determine
        whether the benchmark should be skipped.
    skip_message : str
        The message to display if the bench is skipped.

    Returns
    -------
    function
        The custom skip decorator.
    """
    def decorator(func):
        if (isinstance(condition, bool) and condition) or \
                (not isinstance(condition, bool) and condition()):
            # display a message and skip bench
            wrapper = unittest.skip("[{}] skipped: {}\n".format(_get_bench_name(func), skip_message))(func)
        else:
            # allow execution
            @wraps(func)
            def wrapper(*args, **kwargs):
                func(*args, **kwargs)
        return wrapper
    return decorator


def _get_bench_name(bench_func):
    """Get the benchmark name from its function object."""
    return bench_func.__name__[len(BENCH_METHOD_PREFIX) + 1:]


def benchmark(setup=None, number=10, repeat=3, warmup=5):
    """A parametrized decorator to benchmark the test.

    Setting up the bench can happen in the normal setUp,
    which is applied to all benches identically, and additionally
    the setup parameter, which is bench-specific.

    Parameters
    ----------
    setup : function
        A function to call once to set up the test.
    number : int
        The number of loops of repeat repeats to run.
    repeat : int
        The number of repeats in each loop.
    warmup : int
        The number of warmup runs of the function.
    """
    def real_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if setup is not None:
                setup(self)
            for i in range(warmup):
                func(self, *args, **kwargs)
            clock_time_starts = np.zeros((number, repeat))
            clock_time_ends = np.zeros((number, repeat))
            for i in range(number):
                for j in range(repeat):
                    clock_time_starts[i, j] = perf_counter()
                    func(self, *args, **kwargs)
                    clock_time_ends[i, j] = perf_counter()
            clock_times = (clock_time_ends - clock_time_starts).min(axis=1)

            print("[{}] with {} loops, best of {}:"
                  .format(_get_bench_name(func), number, repeat))
            print("\tmin {:4s} per loop".format(_timeitlike_time_format(clock_times.min())))
            print("\tavg {:4s} per loop".format(_timeitlike_time_format(clock_times.mean())))
        return wrapper
    return real_decorator


pandas_only = _bench_skipper(not hasattr(Table, '_metadata'),
                             "Not a pandas environment.")
non_pandas_only = _bench_skipper(hasattr(Table, '_metadata'),
                                 "Not a pre-pandas environment.")


# see Benchmark.setUpClass()
global_setup_ran = False


class Benchmark(unittest.TestCase):
    """A base class for all benchmarks."""

    @classmethod
    def getPlatformSpecificDetails(cls):
        """Get Windows/Linux/OSX-specific details as a string."""
        win = platform.win32_ver()
        lin = platform.linux_distribution()
        osx = platform.mac_ver()
        if win[0]:
            return "{} {} {}".format(*win[:3])
        elif lin[0]:
            return "{} {} {}".format(*lin)
        elif osx[0]:
            return "OSX {} {}".format(osx[0], osx[2])
        else:
            return "no specific system info"

    @classmethod
    def setUpClass(cls):
        """Runs once globally to print system information."""
        global global_setup_ran
        if not global_setup_ran:
            print("\nRunning benchmark with {} v{} on {} ({})"
                  .format(platform.python_implementation(),
                          platform.python_version(),
                          platform.platform(),
                          Benchmark.getPlatformSpecificDetails()))
            global_setup_ran = True
