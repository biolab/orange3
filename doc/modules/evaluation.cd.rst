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
    >>> import matplotlib.pyplot as plt
    >>> names = ["first", "third", "second", "fourth" ]
    >>> avranks =  [1.9, 3.2, 2.8, 3.3 ]
    >>> cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
    >>> Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    >>> plt.show()

The code produces the following graph:

.. image:: /images/statExamples-graph_ranks1.png