"""
=================================
Sampling of examples (``sample``)
=================================

Example sampling is one of the basic procedures in machine learning. If
for nothing else, everybody needs to split dataset into training and
testing examples. 
 
It is easy to select a subset of examples in Orange. The key idea is the
use of indices: first construct a list of indices, one corresponding
to each example. Then you can select examples by indices, say take
all examples with index 3. Or with index other than 3. It is obvious
that this is useful for many typical setups, such as 70-30 splits or
cross-validation. 
 
Orange provides methods for making such selections, such as
:obj:`Orange.data.Table.select`.  And, of course, it provides methods
for constructing indices for different kinds of splits. For instance,
for the most common used sampling method, cross-validation, the Orange's
class :obj:`SubsetIndicesCV` prepares a list of indices that assign a
fold to each example.

Classes that construct such indices are derived from a basic
abstract :obj:`SubsetIndices`. There are three different classes
provided. :obj:`SubsetIndices2` constructs a list of 0's and 1's in
prescribed proportion; it can be used for, for instance, 70-30 divisions
on training and testing examples. A more general :obj:`SubsetIndicesN`
construct a list of indices from 0 to N-1 in given proportions. Finally,
the most often used :obj:`SubsetIndicesCV` prepares indices for
cross-validation.

Subset indices are more deterministic than in versions of Orange prior to
September 2003. See examples in the section about :obj:`SubsetIndices2`
for details.
 
.. class:: SubsetIndices

    .. data:: Stratified

    .. data:: NotStratified

    .. data:: StratifiedIfPossible
        
        Constants for setting :obj:`stratified`. If
        :obj:`StratifiedIfPossible`, Orange will try to construct
        stratified indices, but fall back to non-stratified if anything
        goes wrong. For stratified indices, it needs to see the example
        table (see the calling operator below), and the class should be
        discrete and have no unknown values.


    .. attribute:: stratified

        Defines whether the division should be stratified, that is,
        whether all subset should have approximatelly equal class
        distributions. Possible values are :obj:`Stratified`,
        :obj:`NotStratified` and :obj:`StratifiedIfPossible` (default).

    .. attribute:: randseed
    
    .. attribute:: random_generator

        These two fields deal with the way :obj:`SubsetIndices` generates
        random numbers.

        If :obj:`random_generator` (of type :obj:`Orange.misc.Random`)
        is set, it is used. The same random generator can be shared
        between different objects; this can be useful when constructing an
        experiment that depends on a single random seed. If you use this,
        :obj:`SubsetIndices` will return a different set of indices each
        time it's called, even if with the same arguments.

        If :obj:`random_generator` is not given, but :attr:`randseed` is
        (positive values denote a defined :obj:`randseed`), the value is
        used to initiate a new, temporary local random generator. This
        way, the indices generator will always give same indices for
        the same data.

        If none of the two is defined, a new random generator
        is constructed each time the object is called (note that
        this is unlike some other classes, such as :obj:`Variable`,
        :obj:`Distribution` and :obj:`Orange.data.Table`, that store
        such generators for future use; the generator constructed by
        :obj:`SubsetIndices` is disposed after use) and initialized
        with random seed 0. This thus has the same effect as setting
        :obj:`randseed` to 0.

        The example for :obj:`SubsetIndices2` shows the difference
        between those options.

    .. method:: __call__(examples)

        :obj:`SubsetIndices` can be called to return a list of
        indices. The argument can be either the desired length of the list
        (presumably corresponding to a length of some list of examples)
        or a set of examples, given as :obj:`Orange.data.Table` or plain
        Python list. It is obvious that in the former case, indices
        cannot correspond to a stratified division; if :obj:`stratified`
        is set to :obj:`Stratified`, an exception is raised.

.. class:: SubsetIndices2

    This object prepares a list of 0's and 1's.
 
    .. attribute:: p0

        The proportion or a number of 0's. If :obj:`p0` is less than
        1, it's a proportion. For instance, if :obj:`p0` is 0.2, 20%
        of indices will be 0's and 80% will be 1's. If :obj:`p0`
        is 1 or more, it gives the exact number of 0's. For instance,
        with :obj:`p0` of 10, you will get a list with 10 0's and
        the rest of the list will be 1's.
 
Say that you have loaded the lenses domain into ``data``. We'll split
it into two datasets, the first containing only 6 examples and the other
containing the rest (from :download:`randomindices2.py <code/randomindices2.py>`):
 
.. literalinclude:: code/randomindices2.py
    :lines: 11-17

Output::

    <1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1>
    6 18
 
No surprises here. Let's now see what's with those random seeds and generators. First, we shall simply construct and print five lists of random indices. 
 
.. literalinclude:: code/randomindices2.py
    :lines: 19-21

Output::

    Indices without playing with random generator
    <0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
    <0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
    <0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
    <0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>
    <0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1>


We ran it for five times and got the same result each time.

.. literalinclude:: code/randomindices2.py
    :lines: 23-26

Output::

    Indices with random generator
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
    <1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1>
    <1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1>
    <1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0>
    <1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1>

We have constructed a private random generator for random indices. And
got five different lists but if you run the whole script again, you'll
get the same five sets, since the generator will be constructed again
and start generating number from the beginning. Again, you should have
got this same indices on any operating system.

.. literalinclude:: code/randomindices2.py
    :lines: 28-32

Output::

    Indices with randseed
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>


Here we have set the random seed and removed the random generator
(otherwise the seed would have no effect as the generator has the
priority). Each time we run the indices generator, it constructs a
private random generator and initializes it with the given seed, and
consequentially always returns the same indices.

Let's play with :obj:`SubsetIndices2.p0`. There are 24 examples in the
dataset. Setting :obj:`SubsetIndices2.p0` to 0.25 instead of 6 shouldn't
alter the indices. Let's check it.

.. literalinclude:: code/randomindices2.py
    :lines: 35-37

Output::

    Indices with p0 set as probability (not 'a number of')
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>

Finally, let's observe the effects of :obj:`~SubsetIndices.stratified`. By
default, indices are stratified if it's possible and, in our case,
it is and they are.

.. literalinclude:: code/randomindices2.py
    :lines: 39-49

Output::

    ... with stratification
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>
    <0.625, 0.167, 0.208>
    <0.611, 0.167, 0.222>

We explicitly requested stratication and got the same indices as
before. That's OK. We also printed out the distribution for the whole
dataset and for the selected dataset (as we gave no second parameter,
the examples with no-null indices got selected). They are not same, but
they are pretty close. :obj:`SubsetIndices2` did what it could. Now let's
try without stratification. The script is pretty same except for changing
:obj:`~SubsetIndices.stratified` to :obj:`~SubsetIndices.NotStratified`.

.. literalinclude:: code/randomindices2.py
    :lines: 51-62

Output::
    
    ... and without stratification
    <0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1>
    <0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1>
    <0.625, 0.167, 0.208>
    <0.611, 0.167, 0.222>


Different indices and ... just look at the distribution. Could be worse
but, well, :obj:`~SubsetIndices.NotStratified` doesn't mean that Orange
will make an effort to get uneven distributions. It just won't mind
about them.

For a final test, you can set the class of one of the examples to unknown
and rerun the last script with setting :obj:`~SubsetIndices.stratified`
once to :obj:`~SubsetIndices.Stratified` and once to
:obj:`~SubsetIndices.StratifiedIfPossible`. In the first case you'll
get an error and in the second you'll have a non-stratified indices.

.. literalinclude:: code/randomindices2.py
    :lines: 64-70

Output::

    ... stratified 'if possible'
    <1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1>

    ... stratified 'if possible', after removing the first example's class
    <0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1>
 
.. class:: SubsetIndicesN

    A straight generalization of :obj:`RandomIndices2`, so there's not
    much to be told about it.

    .. attribute:: p

        A list of proportions of examples that go to each fold. If
        :obj:`p` has a length of 3, the returned list will have four
        different indices, the first three will have probabilities as
        defined in :obj:`p` while the last will have a probability of
        (1 - sum of elements of :obj:`p`).

:obj:`SubsetIndicesN` does not support stratification; setting
:obj:`stratified` to :obj:`Stratified` will yield an error.

Let us construct a list of indices that would assign half of examples
to the first set and a quarter to the second and third (part of
:download:`randomindicesn.py <code/randomindicesn.py>`):

.. literalinclude:: code/randomindicesn.py
    :lines: 9-14

Output:

    <1, 0, 0, 2, 0, 1, 1, 0, 2, 0, 2, 2, 1, 0, 0, 0, 2, 0, 0, 0, 1, 2, 1, 0>

Count them and you'll see there are 12 zero's and 6 one's and two's out of 24.
 
.. class:: SubsetIndicesCV
 
    :obj:`SubsetIndicesCV` computes indices for cross-validation.

    It constructs a list of indices between 0 and :obj:`folds` -1
    (inclusive), with an equal number of each (if the number of examples
    is not divisible by :obj:`folds`, the last folds will have one
    example less).

    .. attribute:: folds

        Number of folds. Default is 10.
 
We shall prepare indices for an ordinary ten-fold cross validation and
indices for 10 examples for 5-fold cross validation. For the latter,
we shall only pass the number of examples, which, of course, prevents
the stratification. Part of :download:`randomindicescv.py <code/randomindicescv.py>`):

.. literalinclude:: code/randomindicescv.py
    :lines: 7-12

Output::

    Indices for ordinary 10-fold CV
    <1, 1, 3, 8, 8, 3, 2, 7, 5, 0, 1, 5, 2, 9, 4, 7, 4, 9, 3, 6, 0, 2, 0, 6>
    Indices for 5 folds on 10 examples
    <3, 0, 1, 0, 3, 2, 4, 4, 1, 2>


Since examples don't divide evenly into ten folds, the first four folds
have one example more - there are three 0's, 1's, 2's and 3's, but only
two 4's, 5's..

"""

pass

from orange import \
     MakeRandomIndices as SubsetIndices, \
     MakeRandomIndicesN as SubsetIndicesN, \
     MakeRandomIndicesCV as SubsetIndicesCV, \
     MakeRandomIndicesMultiple as SubsetIndicesMultiple, \
     MakeRandomIndices2 as SubsetIndices2
