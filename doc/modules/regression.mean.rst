.. py:currentmodule:: Orange.regression.mean

###############
Mean (``mean``)
###############

.. index:: mean fitter
   pair: regression; mean fitter

*Mean model* predicts the same value (usually the distribution mean) for all
data instances. Its accuracy can serve as a baseline for other regression
models.

The model fitter (:class:`MeanFitter`) computes the mean of the given data or
distribution. The model is stored as an instance of :class:`MeanModel`. ::

    >>> from Orange.data import Table
    >>> from Orange.regression.mean import MeanFitter
    >>> mpg = Table('auto-mpg')
    >>> fitter = MeanFitter()
    >>> model = fitter(mpg)
    >>> print(model)
    MeanModel(23.51457286432161)
    >>> model(mpg[:4])
    array([ 23.51457286,  23.51457286,  23.51457286,  23.51457286])

.. autoclass:: MeanFitter
   :members:

.. autoclass:: MeanModel
   :members:
