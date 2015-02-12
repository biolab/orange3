.. py:currentmodule:: Orange.evaluation.scoring

########################
CD diagram (``scoring``)
########################

.. index:: CD diagram

.. autofunction:: Orange.evaluation.compute_CD
.. autofunction:: Orange.evaluation.graph_ranks

Example
=======

    >>> import Orange
    >>> names = ["first", "third", "second", "fourth" ]
    >>> avranks =  [1.9, 3.2, 2.8, 3.3 ]
    >>> cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
    >>> fig = Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    >>> fig.show()

The code produces the following graph:

.. image:: /images/statExamples-graph_ranks1.png