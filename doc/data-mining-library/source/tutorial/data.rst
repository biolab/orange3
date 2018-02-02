The Data
========

.. index: data

This section describes how to load the data in Orange. We also show how to explore the data, perform some basic statistics, and how to sample the data.

Data Input
----------

..  index:: 
    single: data; input

Orange can read files in native tab-delimited format, or can load data from any of the major standard spreadsheet file type, like CSV and Excel. Native format starts with a header row with feature (column) names. Second header row gives the attribute type, which can be continuous, discrete, time, or string. The third header line contains meta information to identify dependent features (class), irrelevant features (ignore) or meta features (meta).
More detailed specification is available in :doc:`../reference/data.io`.
Here are the first few lines from a dataset :download:`lenses.tab <code/lenses.tab>`::

    age       prescription  astigmatic    tear_rate     lenses
    discrete  discrete      discrete      discrete      discrete 
                                                        class
    young     myope         no            reduced       none
    young     myope         no            normal        soft
    young     myope         yes           reduced       none
    young     myope         yes           normal        hard
    young     hypermetrope  no            reduced       none


Values are tab-limited. This dataset has four attributes (age of the patient, spectacle prescription, notion on astigmatism, and information on tear production rate) and an associated three-valued dependent variable encoding lens prescription for the patient (hard contact lenses, soft contact lenses, no lenses). Feature descriptions could use one letter only, so the header of this dataset could also read::

    age       prescription  astigmatic    tear_rate     lenses
    d         d             d             d             d 
                                                        c

The rest of the table gives the data. Note that there are 5 instances in our table above. For the full dataset, check out or download :download:`lenses.tab <code/lenses.tab>`) to a target directory. You can also skip this step as Orange comes preloaded with several demo datasets, lenses being one of them. Now, open a python shell, import Orange and load the data:

    >>> import Orange
    >>> data = Orange.data.Table("lenses")
    >>>

Note that for the file name no suffix is needed; as Orange checks if any files in the current directory are of the readable type. The call to ``Orange.data.Table`` creates an object called ``data`` that holds your dataset and information about the lenses domain:

    >>> data.domain.attributes
    (DiscreteVariable('age', values=['pre-presbyopic', 'presbyopic', 'young']),
     DiscreteVariable('prescription', values=['hypermetrope', 'myope']),
     DiscreteVariable('astigmatic', values=['no', 'yes']),
     DiscreteVariable('tear_rate', values=['normal', 'reduced']))
    >>> data.domain.class_var
    DiscreteVariable('lenses', values=['hard', 'none', 'soft'])
    >>> for d in data[:3]:
       ...:     print(d)
       ...:
    [young, myope, no, reduced | none]
    [young, myope, no, normal | soft]
    [young, myope, yes, reduced | none]
    >>>

The following script wraps-up everything we have done so far and lists first 5 data instances with ``soft`` perscription:

.. literalinclude:: code/data-lenses.py

Note that data is an object that holds both the data and information on the domain. We show above how to access attribute and class names, but there is much more information there, including that on feature type, set of values for categorical features, and other.

Saving the Data
---------------

Data objects can be saved to a file:

    >>> data.save("new_data.tab")
    >>>

This time, we have to provide the file extension to specify the output format. An extension for native Orange's data format is ".tab". The following code saves only the data items with myope perscription:

.. literalinclude:: code/data-save.py

We have created a new data table by passing the information on the structure of the data (``data.domain``) and a subset of data instances.

Exploration of the Data Domain
------------------------------

..  index::
    single: data; attributes
..  index::
    single: data; domain
..  index::
    single: data; class

Data table stores information on data instances as well as on data domain. Domain holds the names of attributes, optional classes, their types and, and if categorical, the value names. The following code:

..  literalinclude:: code/data-domain1.py

outputs::

    25 attributes: 14 continuous, 11 discrete
    First three attributes: symboling, normalized-losses, make
    Class: price

Orange's objects often behave like Python lists and dictionaries, and can be indexed or accessed through feature names:

..  literalinclude:: code/data-domain2.py
    :lines: 5-

The output of the above code is::

    First attribute: symboling
    Values of attribute 'fuel-type': diesel, gas

Data Instances
--------------

..  index::
    single: data; instances
..  index::
    single: data; examples

Data table stores data instances (or examples). These can be index or traversed as any Python list. Data instances can be considered as vectors, accessed through element index, or through feature name.

..  literalinclude:: code/data-instances1.py

The script above displays the following output::

    First three data instances:
    [5.100, 3.500, 1.400, 0.200 | Iris-setosa]
    [4.900, 3.000, 1.400, 0.200 | Iris-setosa]
    [4.700, 3.200, 1.300, 0.200 | Iris-setosa]
    25-th data instance:
    [4.800, 3.400, 1.900, 0.200 | Iris-setosa]
    Value of 'sepal width' for the first instance: 3.500
    The 3rd value of the 25th data instance: 1.900

Iris dataset we have used above has four continous attributes. Here's a script that computes their mean:

..  literalinclude:: code/data-instances2.py
    :lines: 3-

Above also illustrates indexing of data instances with objects that store features; in ``d[x]`` variable ``x`` is an Orange object. Here's the output::

    Feature         Mean
    sepal length    5.84
    sepal width     3.05
    petal length    3.76
    petal width     1.20


A slightly more complicated, but more interesting is a code that computes per-class averages:

..  literalinclude:: code/data-instances3.py
    :lines: 3-

Of the four features, petal width and length look quite discriminative for the type of iris:

    Feature             Iris-setosa Iris-versicolor  Iris-virginica
    sepal length               5.01            5.94            6.59
    sepal width                3.42            2.77            2.97
    petal length               1.46            4.26            5.55
    petal width                0.24            1.33            2.03

Finally, here is a quick code that computes the class distribution for another dataset:

..  literalinclude:: code/data-instances4.py

Orange Datasets and NumPy
-------------------------
Orange datasets are actually wrapped `NumPy <http://www.numpy.org>`_ arrays. Wrapping is performed to retain the information about the feature names and values, and NumPy arrays are used for speed and compatibility with different machine learning toolboxes, like `scikit-learn <http://scikit-learn.org>`_, on which Orange relies. Let us display the values of these arrays for the first three data instances of the iris dataset::

    >>> data = Orange.data.Table("iris")
    >>> data.X[:3]
    array([[ 5.1,  3.5,  1.4,  0.2],
           [ 4.9,  3. ,  1.4,  0.2],
           [ 4.7,  3.2,  1.3,  0.2]])
    >>> data.Y[:3]
    array([ 0.,  0.,  0.])

Notice that we access the arrays for attributes and class separately, using ``data.X`` and ``data.Y``. Average values of attributes can then be computed efficiently by::

    >>> import np as numpy
    >>> np.mean(data.X, axis=0)
    array([ 5.84333333,  3.054     ,  3.75866667,  1.19866667])

We can also construct a (classless) dataset from a numpy array::

    >>> X = np.array([[1,2], [4,5]])
    >>> data = Orange.data.Table(X)
    >>> data.domain
    [Feature 1, Feature 2]

If we want to provide meaninful names to attributes, we need to construct an appropriate data domain::

    >>> domain = Orange.data.Domain([Orange.data.ContinuousVariable("lenght"), 
                                     Orange.data.ContinuousVariable("width")])
    >>> data = Orange.data.Table(domain, X)
    >>> data.domain
    [lenght, width]

Here is another example, this time with construction of dataset that includes a numerical class and different type of attributes:

..  literalinclude:: code/data-domain-numpy.py
    :lines: 4-

Running of this scripts yields::

    [[big, 3.400, circle | 42.000],
     [small, 2.700, oval | 52.200],
     [big, 1.400, square | 13.400]

Meta Attributes
---------------

Often, we wish to include descriptive fields in the data that will not be used in any computation (distance estimation, modeling), but will serve for identification or additional information. These are called meta attributes, and are marked with ``meta`` in the third header row:

..  literalinclude:: code/zoo.tab

Values of meta attributes and all other (non-meta) attributes are treated similarly in Orange, but stored in the separate numpy arrays:

    >>> data = Orange.data.Table("zoo")
    >>> data[0]["name"]
    >>> data[0]["type"]
    >>> for d in data:
        ...:     print("{}/{}: {}".format(d["name"], d["type"], d["legs"]))
        ...:
    aardvark/mammal: 4
    antelope/mammal: 4
    bass/fish: 0
    bear/mammal: 4
    >>> data.X
    array([[ 1.,  0.,  1.,  1.,  2.],
           [ 1.,  0.,  1.,  1.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 1.,  0.,  1.,  1.,  2.]]))
    >>> data.metas
    array([['aardvark'],
           ['antelope'],
           ['bass'],
           ['bear']], dtype=object))

Meta attributes may be passed to ``Orange.data.Table`` after providing arrays for attribute and class values:

..   literalinclude:: code/data-metas.py

The script outputs::

    [[2.200, 1625.000 | no] {houston, 10},
     [0.300, 163.000 | yes] {ljubljana, -1}

To construct a classless domain we could pass ``None`` for the class values.

Missing Values
--------------

..  index::
    single: data; missing values

Consider the following exploration of the dataset on votes of the US senate::

    >>> import numpy as np
    >>> data = Orange.data.Table("voting.tab")
    >>> data[2]
    [?, y, y, ?, y, ... | democrat]
    >>> np.isnan(data[2][0])
    True
    >>> np.isnan(data[2][1])
    False

The particular data instance included missing data (represented with '?') for the first and the fourth attribute. In the original dataset file, the missing values are, by default, represented with a blank space. We can now examine each attribute and report on proportion of data instances for which this feature was undefined:

..  literalinclude:: code/data-missing.py
    :lines: 4-

First three lines of the output of this script are::

     2.8% handicapped-infants
    11.0% water-project-cost-sharing
     2.5% adoption-of-the-budget-resolution

A single-liner that reports on number of data instances with at least one missing value is::

    >>> sum(any(np.isnan(d[x]) for x in data.domain.attributes) for d in data)
    203 

.. sum([np.any(np.isnan(x)) for x in data.X])

Data Selection and Sampling
---------------------------

..  index::
    single: data; sampling

Besides the name of the data file, ``Orange.data.Table`` can accept the data domain and a list of data items and returns a new dataset. This is useful for any data subsetting:

..  literalinclude:: code/data-subsetting.py
    :lines: 3-

The code outputs::

    Dataset instances: 150
    Subset size: 99

and inherits the data description (domain) from the original dataset. Changing the domain requires setting up a new domain descriptor. This feature is useful for any kind of feature selection:

..  literalinclude:: code/data-feature-selection.py
    :lines: 3-

..  index::
    single: feature; selection

We could also construct a random sample of the dataset::

    >>> sample = Orange.data.Table(data.domain, random.sample(data, 3))
    >>> sample
    [[6.000, 2.200, 4.000, 1.000 | Iris-versicolor],
     [4.800, 3.100, 1.600, 0.200 | Iris-setosa],
     [6.300, 3.400, 5.600, 2.400 | Iris-virginica]
    ]

or randomly sample the attributes:

    >>> atts = random.sample(data.domain.attributes, 2)
    >>> domain = Orange.data.Domain(atts, data.domain.class_var)
    >>> new_data = Orange.data.Table(domain, data)
    >>> new_data[0]
    [5.100, 1.400 | Iris-setosa]
