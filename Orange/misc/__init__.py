"""
.. index:: misc

.. index: CostMatrix

-----------------------
CostMatrix
-----------------------

CostMatrix is an object that stores costs of (mis)classifications. Costs can be either negative or positive.

.. class:: CostMatrix

    .. attribute:: class_var 
        
        The (class) attribute to which the matrix applies. This can
        also be None.
        
    .. attribute:: dimension (read only)
    
        Matrix dimension, ie. number of classes.
        
    .. method:: CostMatrix(dimension[, default cost])
    
        Constructs a matrix of the given size and initializes it with
        the default cost (1, if not given). All elements of the matrix
        are assigned the given cost, except for the diagonal that have
        the default cost of 0.  (Diagonal elements represent correct
        classifications and these usually have no price; you can,
        however, change this.)
        
        .. literalinclude:: code/CostMatrix.py
            :lines: 1-8
        
        This initializes the matrix and print it out:
        
        .. literalinclude:: code/CostMatrix.res
            :lines: 1-3
    
    .. method:: CostMatrix(class descriptor[, default cost])
    
        Similar as above, except that classVar is also set to the given descriptor.
        The number of values of the given attribute (which must be discrete) is used
        for dimension.
        
        .. literalinclude:: code/CostMatrix.py
            :lines: 10-11
            
        This constructs a matrix similar to the one above (the class attribute in iris
        domain is three-valued) except that the matrix contains 2s instead of 1s.
        
    .. method:: CostMatrix([attribute descriptor, ]matrix)
    
        Initializes the matrix with the elements given as a sequence of sequences (you
        can mix lists and tuples if you find it funny). Each subsequence represents a row.
        
        .. literalinclude:: code/CostMatrix.py
            :lines: 13

        If you print this matrix out, will it look like this:
        
        .. literalinclude:: code/CostMatrix.res
            :lines: 5-7
            
    .. method:: setcost(predicted, correct, cost)
    
        Set the misclassification cost. The matrix above could be
        constructed by first initializing it with 2s and then changing
        the prices for virginica's into 1s.
        
        .. literalinclude:: code/CostMatrix.py
            :lines: 15-17
            
    .. method:: getcost(predicted, correct)
    
        Returns the cost of prediction. Values must be integer
        indices; if class_var is set, you can also use symbolic values
        (strings). Note that there's no way to change the size of the
        matrix. Size is set at construction and does not change.  For
        the final example, we shall compute the profits of knowing
        attribute values in the dataset lenses with the same
        cost-matrix as printed above.
        
        .. literalinclude:: code/CostMatrix.py
            :lines: 19-23
            
        As the script shows, you don't have to (and usually won't) call the constructor
        explicitly. Instead, you will set the corresponding field (in our case meas.cost)
        to a matrix and let Orange convert it to CostMatrix automatically. Funny as it
        might look, but since Orange uses constructor to perform such conversion, even
        the above statement is correct (although the cost matrix is rather dull,
        with 0s on the diagonal and 1s around):            
            
        .. literalinclude:: code/CostMatrix.py
            :lines: 25
                
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
	
    .. method:: __init__(dim[, value])

        Construct a symmetric matrix of the given dimension.

        :param dim: matrix dimension
        :type dim: int

        :param value: default value (0 by default)
        :type value: double
        
        
    .. method:: __init__(data)

        Construct a new symmetric matrix containing the given data. 
        These can be given as Python list containing lists or tuples.
        
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
        
        

Indexing
..........

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


.. class:: Random(seed)

    :param initseed: Seed used for initializing the random generator.
    :type initseed: int

    .. method:: __call__(n)

        Return a random integer R such that 0 <= R < n.

        :type n: int

    .. method:: reset([seed])

        Reinitialize the random generator with `initseed`. If `initseed`
        is not given use the existing value of attribute `initseed`.

    .. attribute:: uses
        
        The number of times the generator was called after
        initialization/reset.
    
    .. attribute:: initseed

        Random seed.

Two examples or random number generator uses found in the documentation
are :obj:`Orange.evaluation.testing` and :obj:`Orange.data.Table`.

"""
from functools import wraps
from Orange.core import RandomGenerator as Random
from Orange.core import SymMatrix
from Orange.core import CostMatrix
