"""
***********************
Majority (``majority``)
***********************

.. index:: majority classifier
   pair: classification; majority classifier

Accuracy of classifiers is often compared to the "default accuracy",
that is, the accuracy of a classifier which classifies all instances
to the majority class. To fit into the standard schema, even this
algorithm is provided in form of the usual learner-classifier pair.
Learning is done by :obj:`MajorityLearner` and the classifier it
constructs is an instance of :obj:`ConstantClassifier`.

.. class:: MajorityLearner

    MajorityLearner will most often be used as is, without setting any
    parameters. Nevertheless, it has two.

    .. attribute:: estimator_constructor
    
        An estimator constructor that can be used for estimation of
        class probabilities. If left None, probability of each class is
        estimated as the relative frequency of instances belonging to
        this class.
        
    .. attribute:: apriori_distribution
    
        Apriori class distribution that is passed to estimator
        constructor if one is given.

.. class:: ConstantClassifier

    ConstantClassifier always classifies to the same class and reports the
    same class probabilities.

    Its constructor can be called without arguments, with a variable (for
    :obj:`class_var`), value (for :obj:`default_val`) or both. If the value
    is given and is of type :obj:`Orange.data.Value` (alternatives are an
    integer index of a discrete value or a continuous value), its attribute
    :obj:`Orange.data.Value.variable` will either be used for initializing
    :obj:`class_var` if variable is not given as an argument, or checked
    against the variable argument, if it is given. 
    
    .. attribute:: default_val
    
        Value that is returned by the classifier.
    
    .. attribute:: default_distribution

        Class probabilities returned by the classifier.
    
    .. attribute:: class_var
    
        Class variable that the classifier predicts.


Examples
========

This "learning algorithm" will most often be used as a baseline,
that is, to determine if some other learning algorithm provides
any information about the class (:download:`majority-classification.py <code/majority-classification.py>`,
uses: :download:`monks-1.tab <code/monks-1.tab>`):

.. literalinclude:: code/majority-classification.py
    :lines: 7-

"""

from Orange.core import MajorityLearner
from Orange.core import DefaultClassifier as ConstantClassifier
