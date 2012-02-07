"""
.. index:: misc

Module Orange.misc contains common functions and classes which are used in other modules.

.. index: SymMatrix

-----------------------
SymMatrix
-----------------------

:obj:`SymMatrix` implements symmetric matrices of size fixed at 
construction time (and stored in :obj:`SymMatrix.dim`).

.. class:: SymMatrix

    .. attribute:: dim
	
        Matrix dimension.
            
    .. attribute:: matrix_type 

        Can be ``SymMatrix.Lower`` (0), ``SymMatrix.Upper`` (1), 
        ``SymMatrix.Symmetric`` (2, default), ``SymMatrix.LowerFilled`` (3) or
        ``SymMatrix.Upper_Filled`` (4). 

        If the matrix type is ``Lower`` or ``Upper``, indexing 
        above or below the diagonal, respectively, will fail. 
        With ``LowerFilled`` and ``Upper_Filled``,
        the elements upper or lower, respectively, still 
        exist and are set to zero, but they cannot be modified. The 
        default matrix type is ``Symmetric``, but can be changed 
        at any time.

        If matrix type is ``Upper``, it is printed as:

        >>> import Orange
        >>> m = Orange.misc.SymMatrix(
        ...     [[1], 
        ...      [2, 4], 
        ...      [3, 6, 9], 
        ...      [4, 8, 12, 16]])
        >>> m.matrix_type = m.Upper
        >>> print m
        (( 1.000,  2.000,  3.000,  4.000),
         (         4.000,  6.000,  8.000),
         (                 9.000, 12.000),
         (                        16.000))

        Changing the type to ``LowerFilled`` changes the printout to

        >>> m.matrix_type = m.LowerFilled
        >>> print m
        (( 1.000,  0.000,  0.000,  0.000),
         ( 2.000,  4.000,  0.000,  0.000),
         ( 3.000,  6.000,  9.000,  0.000),
         ( 4.000,  8.000, 12.000, 16.000))
	
    .. method:: __init__(dim[, default_value])

        Construct a symmetric matrix of the given dimension.

        :param dim: matrix dimension
        :type dim: int

        :param default_value: default value (0 by default)
        :type default_value: double
        
        
    .. method:: __init__(instances)

        Construct a new symmetric matrix containing the given data instances. 
        These can be given as Python list containing lists or tuples.

        :param instances: data instances
        :type instances: list of lists
        
        The following example fills a matrix created above with
        data in a list::

            import Orange
            m = [[],
                 [ 3],
                 [ 2, 4],
                 [17, 5, 4],
                 [ 2, 8, 3, 8],
                 [ 7, 5, 10, 11, 2],
                 [ 8, 4, 1, 5, 11, 13],
                 [ 4, 7, 12, 8, 10, 1, 5],
                 [13, 9, 14, 15, 7, 8, 4, 6],
                 [12, 10, 11, 15, 2, 5, 7, 3, 1]]
                    
            matrix = Orange.data.SymMatrix(m)

        SymMatrix also stores diagonal elements. They are set
        to zero, if they are not specified. The missing elements
        (shorter lists) are set to zero as well. If a list
        spreads over the diagonal, the constructor checks
        for asymmetries. For instance, the matrix

        ::

            m = [[],
                 [ 3,  0, f],
                 [ 2,  4]]
    
        is only OK if f equals 2. Finally, no row can be longer 
        than matrix size.  

    .. method:: get_values()
    
        Return all matrix values in a Python list.

    .. method:: get_KNN(i, k)
    
        Return k columns with the lowest value in the i-th row. 
        
        :param i: i-th row
        :type i: int
        
        :param k: number of neighbors
        :type k: int
        
    .. method:: avg_linkage(clusters)
    
        Return a symmetric matrix with average distances between given clusters.  
      
        :param clusters: list of clusters
        :type clusters: list of lists
        
    .. method:: invert(type)
    
        Invert values in the symmetric matrix.
        
        :param type: 0 (-X), 1 (1 - X), 2 (max - X), 3 (1 / X)
        :type type: int

    .. method:: normalize(type)
    
        Normalize values in the symmetric matrix.
        
        :param type: 0 (normalize to [0, 1] interval), 1 (Sigmoid)
        :type type: int
        
        
-------------------
Indexing
-------------------

For symmetric matrices the order of indices is not important: 
if ``m`` is a SymMatrix, then ``m[2, 4]`` addresses the same element as ``m[4, 2]``.

..
    .. literalinclude:: code/symmatrix.py
        :lines: 1-6

>>> import Orange
>>> m = Orange.misc.SymMatrix(4)
>>> for i in range(4):
...    for j in range(i+1):
...        m[i, j] = (i+1)*(j+1)


Although only the lower left half of the matrix was set explicitely, 
the whole matrix is constructed.

>>> print m
(( 1.000,  2.000,  3.000,  4.000),
 ( 2.000,  4.000,  6.000,  8.000),
 ( 3.000,  6.000,  9.000, 12.000),
 ( 4.000,  8.000, 12.000, 16.000))
 
Entire rows are indexed with a single index. They can be iterated
over in a for loop or sliced (with, for example, ``m[:3]``):

>>> print m[1]
(2.0, 4.0, 6.0, 8.0)
>>> m.matrix_type = m.Lower
>>> for row in m:
...     print row
(1.0,)
(2.0, 4.0)
(3.0, 6.0, 9.0)
(4.0, 8.0, 12.0, 16.0)

.. index: Random number generator

-----------------------
Random number generator
-----------------------

:obj:`Random` uses the 
`Mersenne twister <http://en.wikipedia.org/wiki/Mersenne_twister>`_ algorithm
to generate random numbers.

::

    >>> import Orange
    >>> rg = Orange.misc.Random(42)
    >>> rg(10)
    4
    >>> rg(10)
    7
    >>> rg.uses  # We called rg two times.
    2
    >>> rg.reset()
    >>> rg(10)
    4
    >>> rg(10)
    7
    >>> rg.uses
    2


.. class:: Random(initseed)

    :param initseed: Seed used for initializing the random generator.
    :type initseed: int

    .. method:: __call__(n)

        Return a random integer R such that 0 <= R < n.

        :type n: int

    .. method:: reset([initseed])

        Reinitialize the random generator with `initseed`. If `initseed`
        is not given use the existing value of attribute `initseed`.

    .. attribute:: uses
        
        The number of times the generator was called after
        initialization/reset.
    
    .. attribute:: initseed

        Random seed.

Two examples or random number generator uses found in the documentation
are :obj:`Orange.evaluation.testing` and :obj:`Orange.data.Table`.

.. automodule:: Orange.misc.counters
  :members:

.. automodule:: Orange.misc.render
  :members:

.. automodule:: Orange.misc.selection

.. automodule:: Orange.misc.addons

.. automodule:: Orange.misc.serverfiles

.. automodule:: Orange.misc.environ

.. automodule:: Orange.misc.r


"""
import environ
import counters
import render

from Orange.core import RandomGenerator as Random
from orange import SymMatrix

# addons is intentionally not imported; if it were, add-ons' directories would
# be added to the python path. If that sounds OK, this can be changed ...

__all__ = ["counters", "selection", "render", "serverfiles",
           "deprecated_members", "deprecated_keywords",
           "deprecated_attribute", "deprecation_warning"]

import random, types, sys
import time

def getobjectname(x, default=""):
    if type(x)==types.StringType:
        return x
      
    for i in ["name", "shortDescription", "description", "func_doc", "func_name"]:
        if getattr(x, i, ""):
            return getattr(x, i)

    if hasattr(x, "__class__"):
        r = repr(x.__class__)
        if r[1:5]=="type":
            return str(x.__class__)[7:-2]
        elif r[1:6]=="class":
            return str(x.__class__)[8:-2]
    return default


def demangle_examples(x):
    if type(x)==types.TupleType:
        return x
    else:
        return x, 0


def frange(*argw):
    """ Like builtin `range` but works with floats
    """
    start, stop, step = 0.0, 1.0, 0.1
    if len(argw)==1:
        start=step=argw[0]
    elif len(argw)==2:
        stop, step = argw
    elif len(argw)==3:
        start, stop, step = argw
    elif len(argw)>3:
        raise AttributeError, "1-3 arguments expected"

    stop+=1e-10
    i=0
    res=[]
    while 1:
        f=start+i*step
        if f>stop:
            break
        res.append(f)
        i+=1
    return res

verbose = 0

def print_verbose(text, *verb):
    if len(verb) and verb[0] or verbose:
        print text

__doc__ += """\
------------------
Reporting progress
------------------

.. autoclass:: Orange.misc.ConsoleProgressBar
    :members:

"""

class ConsoleProgressBar(object):
    """ A class to for printing progress bar reports in the console.
    
    Example ::
    
        >>> import sys, time
        >>> progress = ConsoleProgressBar("Example", output=sys.stdout)
        >>> for i in range(100):
        ...    progress.advance()
        ...    # Or progress.set_state(i)
        ...    time.sleep(0.01)
        ...
        ...
        Example ===================================>100%
        
    """
    def __init__(self, title="", charwidth=40, step=1, output=None):
        """ Initialize the progress bar.
        
        :param title: The title for the progress bar.
        :type title: str
        :param charwidth: The maximum progress bar width in characters.
        
            .. todo:: Get the console width from the ``output`` if the
                information can be retrieved. 
                
        :type charwidth: int
        :param step: A default step used if ``advance`` is called without
            any  arguments
        
        :type step: int
        :param output: The output file. If None (default) then ``sys.stderr``
            is used.
            
        :type output: An file like object to print the progress report to.
         
        """
        self.title = title + " "
        self.charwidth = charwidth
        self.step = step
        self.currstring = ""
        self.state = 0
        if output is None:
            output = sys.stderr
        self.output = output

    def clear(self, i=-1):
        """ Clear the current progress line indicator string.
        """
        try:
            if hasattr(self.output, "isatty") and self.output.isatty():
                self.output.write("\b" * (i if i != -1 else len(self.currstring)))
            else:
                self.output.seek(-i if i != -1 else -len(self.currstring), 2)
        except Exception: ## If for some reason we failed 
            self.output.write("\n")

    def getstring(self):
        """ Return the progress indicator string.
        """
        progchar = int(round(float(self.state) * (self.charwidth - 5) / 100.0))
        return self.title + "=" * (progchar) + ">" + " " * (self.charwidth\
            - 5 - progchar) + "%3i" % int(round(self.state)) + "%"

    def printline(self, string):
        """ Print the ``string`` to the output file.
        """
        try:
            self.clear()
            self.output.write(string)
            self.output.flush()
        except Exception:
            pass
        self.currstring = string

    def __call__(self, newstate=None):
        """ Set the ``newstate`` as the current state of the progress bar.
        ``newstate`` must be in the interval [0, 100].
        
        .. note:: ``set_state`` is the prefered way to set a new steate. 
        
        :param newstate: The new state of the progress bar.
        :type newstate: float
         
        """
        if newstate is None:
            self.advance()
        else:
            self.set_state(newstate)
            
    def set_state(self, newstate):
        """ Set the ``newstate`` as the current state of the progress bar.
        ``newstate`` must be in the interval [0, 100]. 
        
        :param newstate: The new state of the progress bar.
        :type newstate: float
        
        """
        if int(newstate) != int(self.state):
            self.state = newstate
            self.printline(self.getstring())
        else:
            self.state = newstate
            
    def advance(self, step=None):
        """ Advance the current state by ``step``. If ``step`` is None use
        the default step as set at class initialization.
          
        """
        if step is None:
            step = self.step
            
        newstate = self.state + step
        self.set_state(newstate)

    def finish(self):
        """ Finish the progress bar (i.e. set the state to 100 and
        print the final newline to the ``output`` file).
        """
        self.__call__(100)
        self.output.write("\n")

def progress_bar_milestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])

def lru_cache(maxsize=100):
    """ A least recently used cache function decorator.
    (Similar to the functools.lru_cache in python 3.2)
    """
    
    def decorating_function(func):
        import functools
        cache = {}
        
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            if key not in cache:
                res = func(*args, **kwargs)
                cache[key] = (time.time(), res)
                if len(cache) > maxsize:
                    key, (_, _) = min(cache.iteritems(), key=lambda item: item[1][0])
                    del cache[key]
            else:
                _, res = cache[key]
                cache[key] = (time.time(), res) # update the time
                
            return res
        
        def clear():
            cache.clear()
        
        wrapped.clear = clear
        wrapped._cache = cache
        
        return wrapped
    return decorating_function

#from Orange.misc.render import contextmanager
from contextlib import contextmanager


@contextmanager
def member_set(obj, name, val):
    """ A context manager that sets member ``name`` on ``obj`` to ``val``
    and restores the previous value on exit. 
    """
    old_val = getattr(obj, name, val)
    setattr(obj, name, val)
    yield
    setattr(obj, name, old_val)
    
    
class recursion_limit(object):
    """ A context manager that sets a new recursion limit. 
    
    """
    def __init__(self, limit=1000):
        self.limit = limit
        
    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setrecursionlimit(self.old_limit)


__doc__ += """\
-----------------------------
Deprecation utility functions
-----------------------------

.. autofunction:: Orange.misc.deprecation_warning

.. autofunction:: Orange.misc.deprecated_members

.. autofunction:: Orange.misc.deprecated_keywords

.. autofunction:: Orange.misc.deprecated_attribute

.. autofunction:: Orange.misc.deprecated_function_name 

"""

import warnings
def deprecation_warning(old, new, stacklevel=-2):
    """ Raise a deprecation warning of an obsolete attribute access.
    
    :param old: Old attribute name (used in warning message).
    :param new: New attribute name (used in warning message).
    
    """
    warnings.warn("'%s' is deprecated. Use '%s' instead!" % (old, new), DeprecationWarning, stacklevel=stacklevel)
   
# We need to get the instancemethod type 
class _Foo():
    def bar(self):
        pass
instancemethod = type(_Foo.bar)
del _Foo

function = type(lambda: None)

class universal_set(set):
    """ A universal set, pretends it contains everything.
    """
    def __contains__(self, value):
        return True
    
from functools import wraps

def deprecated_members(name_map, wrap_methods="all", in_place=True):
    """ Decorate a class with properties for accessing attributes, and methods
    with deprecated names. In addition methods from the `wrap_methods` list
    will be wrapped to receive mapped keyword arguments.
    
    :param name_map: A dictionary mapping old into new names.
    :type name_map: dict
    
    :param wrap_methods: A list of method names to wrap. Wrapped methods will
        be called with mapped keyword arguments (by default all methods will
        be wrapped).
    :type wrap_methods: list
    
    :param in_place: If True the class will be modified in place, otherwise
        it will be subclassed (default True).
    :type in_place: bool
    
    Example ::
            
        >>> class A(object):
        ...     def __init__(self, foo_bar="bar"):
        ...         self.set_foo_bar(foo_bar)
        ...     
        ...     def set_foo_bar(self, foo_bar="bar"):
        ...         self.foo_bar = foo_bar
        ...
        ... A = deprecated_members(
        ... {"fooBar": "foo_bar", 
        ...  "setFooBar":"set_foo_bar"},
        ... wrap_methods=["set_foo_bar", "__init__"])(A)
        ... 
        ...
        >>> a = A(fooBar="foo")
        __main__:1: DeprecationWarning: 'fooBar' is deprecated. Use 'foo_bar' instead!
        >>> print a.fooBar, a.foo_bar
        foo foo
        >>> a.setFooBar("FooBar!")
        __main__:1: DeprecationWarning: 'setFooBar' is deprecated. Use 'set_foo_bar' instead!
        
    .. note:: This decorator does nothing if \
        :obj:`Orange.misc.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return lambda cls: cls
    
    def is_wrapped(method):
        """ Is member method already wrapped.
        """
        if getattr(method, "_deprecate_members_wrapped", False):
            return True
        elif hasattr(method, "im_func"):
            im_func = method.im_func
            return getattr(im_func, "_deprecate_members_wrapped", False)
        else:
            return False
        
    if wrap_methods == "all":
        wrap_methods = universal_set()
    elif not wrap_methods:
        wrap_methods = set()
        
    def wrapper(cls):
        cls_names = {}
        # Create properties for accessing deprecated members
        for old_name, new_name in name_map.items():
            cls_names[old_name] = deprecated_attribute(old_name, new_name)
            
        # wrap member methods to map keyword arguments
        for key, value in cls.__dict__.items():
            if isinstance(value, (instancemethod, function)) \
                and not is_wrapped(value) and key in wrap_methods:
                
                wrapped = deprecated_keywords(name_map)(value)
                wrapped._deprecate_members_wrapped = True # A flag indicating this function already maps keywords
                cls_names[key] = wrapped
        if in_place:
            for key, val in cls_names.items():
                setattr(cls, key, val)
            return cls
        else:
            return type(cls.__name__, (cls,), cls_names)
        
    return wrapper

def deprecated_keywords(name_map):
    """ Deprecates the keyword arguments of the function.
    
    Example ::
    
        >>> @deprecated_keywords({"myArg": "my_arg"})
        ... def my_func(my_arg=None):
        ...     print my_arg
        ...
        ...
        >>> my_func(myArg="Arg")
        __main__:1: DeprecationWarning: 'myArg' is deprecated. Use 'my_arg' instead!
        Arg
        
    .. note:: This decorator does nothing if \
        :obj:`Orange.misc.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return lambda func: func
    
    def decorator(func):
        @wraps(func)
        def wrap_call(*args, **kwargs):
            kwargs = dict(kwargs)
            for name in name_map:
                if name in kwargs:
                    deprecation_warning(name, name_map[name], stacklevel=3)
                    kwargs[name_map[name]] = kwargs[name]
                    del kwargs[name]
            return func(*args, **kwargs)
        return wrap_call
    return decorator

def deprecated_attribute(old_name, new_name):
    """ Return a property object that accesses an attribute named `new_name`
    and raises a deprecation warning when doing so.

    ..

        >>> sys.stderr = sys.stdout
    
    Example ::
    
        >>> class A(object):
        ...     def __init__(self):
        ...         self.my_attr = "123"
        ...     myAttr = deprecated_attribute("myAttr", "my_attr")
        ...
        ...
        >>> a = A()
        >>> print a.myAttr
        ...:1: DeprecationWarning: 'myAttr' is deprecated. Use 'my_attr' instead!
        123
        
    .. note:: This decorator does nothing and returns None if \
        :obj:`Orange.misc.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return None
    
    def fget(self):
        deprecation_warning(old_name, new_name, stacklevel=3)
        return getattr(self, new_name)
    
    def fset(self, value):
        deprecation_warning(old_name, new_name, stacklevel=3)
        setattr(self, new_name, value)
    
    def fdel(self):
        deprecation_warning(old_name, new_name, stacklevel=3)
        delattr(self, new_name)
    
    prop = property(fget, fset, fdel,
                    doc="A deprecated member '%s'. Use '%s' instead." % (old_name, new_name))
    return prop 


def deprecated_function_name(func):
    """ Return a wrapped function that raises an deprecation warning when
    called. This should be used for deprecation of module level function names. 
    
    Example ::
    
        >>> def func_a(arg):
        ...    print "This is func_a  (used to be named funcA) called with", arg
        ...
        ...
        >>> funcA = deprecated_function_name(func_a)
        >>> funcA(None)
          
    
    .. note:: This decorator does nothing and if \
        :obj:`Orange.misc.environ.orange_no_deprecated_members` environment \
        variable is set to `True`.
        
    """
    if environ.orange_no_deprecated_members:
        return func
    
    @wraps(func)
    def wrapped(*args, **kwargs):
        warnings.warn("Deprecated function name. Use %r instead!" % func.__name__,
                      DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapped
    

"""
Some utility functions common to Orange classes.
 
"""

def _orange__new__(base=None):
    """ Return an orange 'schizofrenic' __new__ class method.
    
    :param base: base orange class (default orange.Learner)
    :type base: type
         
    Example::
        class NewOrangeLearner(orange.Learner):
            __new__ = _orange__new(orange.Learner)
        
    """
    if base is None:
        import Orange
        base = Orange.core.Learner
        
    @wraps(base.__new__)
    def _orange__new_wrapped(cls, data=None, **kwargs):
        self = base.__new__(cls, **kwargs)
        if data:
            self.__init__(**kwargs)
            return self.__call__(data)
        else:
            return self
    return _orange__new_wrapped


def _orange__reduce__(self):
    """ A default __reduce__ method for orange types. Assumes the object
    can be reconstructed with the call `constructor(__dict__)` where __dict__
    if the stored (pickled) __dict__ attribute.
    
    Example::
        class NewOrangeType(orange.Learner):
            __reduce__ = _orange__reduce()
    """ 
    return type(self), (), dict(self.__dict__)


demangleExamples = deprecated_function_name(demangle_examples)
progressBarMilestones = deprecated_function_name(progress_bar_milestones)
printVerbose = deprecated_function_name(print_verbose)

# Must be imported after deprecation function definitions
import selection
