.. py:currentmodule:: Orange.data.io

################################
Loading and saving data (``io``)
################################

:obj:`Orange.data.Table` supports loading from several file formats:

* Comma-separated values (\*.csv) file,
* Tab-separated values (\*.tab, \*.tsv) file,
* Excel spreadsheet (\*.xls, \*.xlsx),
* Basket file,
* Python pickle.

In addition, the text-based files (CSV, TSV) can be compressed with gzip,
bzip2 or xz (e.g. \*.csv.gz).


Header Format
=============

The data in CSV, TSV, and Excel files can be described in an extended
three-line header format, or a condensed single-line header format.


Three-line header format
------------------------

A three-line header consists of:

1. **Feature names** on the first line. Feature names can include any combination
   of characters.

2. **Feature types** on the second line. The type is determined automatically,
   or, if set, can be any of the following:

   * ``discrete`` (or ``d``) — imported as :obj:`Orange.data.DiscreteVariable`,
   * a space-separated **list of discrete values**, like "``male female``",
     which will result in :obj:`Orange.data.DiscreteVariable` with those values
     and in that order. If the individual values contain a space character, it
     needs to be escaped (prefixed) with, as common, a backslash ('\\') character.
   * ``continuous`` (or ``c``) — imported as :obj:`Orange.data.ContinuousVariable`,
   * ``string`` (or ``s``, or ``text``) — imported as :obj:`Orange.data.StringVariable`,
   * ``time`` (or ``t``) — imported as :obj:`Orange.data.TimeVariable`, if the
     values parse as `ISO 8601 <https://en.wikipedia.org/wiki/ISO_8601>`_ date/time formats,
   * ``basket`` — used for storing sparse data. More on basket formats in a
     dedicated section.

3. **Flags** (optional) on the third header line. Feature's flag can be empty,
   or it can contain, space-separated, a consistent combination of:

   * ``class`` (or ``c``) — feature will be imported as a class variable.
     Most algorithms expect a single class variable.
   * ``meta`` (or ``m``) — feature will be imported as a meta-attribute, just
     describing the data instance but not actually used for learning,
   * ``weight`` (or ``w``) — the feature marks the weight of examples (in
     algorithms that support weighted examples),
   * ``ignore`` (or ``i``) — feature will not be imported,
   * ``<key>=<value>`` custom attributes.

Example of iris dataset in Orange's three-line format
(:download:`iris.tab <../../../../Orange/datasets/iris.tab>`).

.. literalinclude:: ../../../../Orange/datasets/iris.tab
   :lines: 1-7


Single-line header format
-------------------------

Single-line header consists of feature names prefixed by an optional "``<flags>#``"
string, i.e. flags followed by a hash ('#') sign. The flags can be a consistent
combination of:

* ``c`` for class feature,
* ``i`` for feature to be ignored,
* ``m`` for meta attributes (not used in learning),
* ``C`` for features that are continuous,
* ``D`` for features that are discrete,
* ``T`` for features that represent date and/or time in one of the ISO 8601
  formats,
* ``S`` for string features.

If some (all) names or flags are omitted, the names, types, and flags are
discerned automatically, and correctly (most of the time).


Baskets
=======

Baskets can be used for storing sparse data in tab delimited files. They were
specifically designed for text mining needs. If text mining and sparse data is
not your business, you can skip this section.

Baskets are given as a list of space-separated ``<name>=<value>`` atoms. A
continuous meta attribute named ``<name>`` will be created and added to the domain
as optional if it is not already there. A meta value for that variable will be
added to the example. If the value is 1, you can omit the ``=<value>`` part.

It is not possible to put meta attributes of other types than continuous in the
basket.

A tab delimited file with a basket can look like this::

    K       Ca      b_foo     Ba  y
    c       c       basket    c   c
            meta              i   class
    0.06    8.75    a b a c   0   1
    0.48            b=2 d     0   1
    0.39    7.78              0   1
    0.57    8.22    c=13      0   1

These are the examples read from such a file::

    [0.06, 1], {"Ca":8.75, "a":2.000, "b":1.000, "c":1.000}
    [0.48, 1], {"Ca":?, "b":2.000, "d":1.000}
    [0.39, 1], {"Ca":7.78}
    [0.57, 1], {"Ca":8.22, "c":13.000}

It is recommended to have the basket as the last column, especially if it
contains a lot of data.

Note a few things. The basket column's name, ``b_foo``, is not used. In the first
example, the value of ``a`` is 2 since it appears twice. The ordinary meta
attribute, ``Ca``, appears in all examples, even in those where its value is
undefined. Meta attributes from the basket appear only where they are defined.
This is due to the different nature of these meta attributes: ``Ca`` is required
while the others are optional.  ::

    >>> d.domain.metas()
    {-6: FloatVariable 'd', -22: FloatVariable 'Ca', -5: FloatVariable 'c', -4: FloatVariable 'b', -3: FloatVariable 'a'}

To fully understand all this, you should read the documentation on :ref:`meta
attributes <meta-attributes>` in Domain and on the :ref:`basket file format
<basket-format>` (a simple format that is limited to baskets only).

.. _basket-format:

Basket Format
-------------

Basket files (.basket) are suitable for representing sparse data. Each example
is represented by a line in the file. The line is written as a comma-separated
list of name-value pairs. Here's an example of such file. ::

    nobody, expects, the, Spanish, Inquisition=5
    our, chief, weapon, is, surprise=3, surprise=2, and, fear,fear, and, surprise
    our, two, weapons, are, fear, and, surprise, and, ruthless, efficiency
    to, the, Pope, and, nice, red, uniforms, oh damn

The file contains four examples. The first examples has five attributes
defined, "nobody", "expects", "the", "Spanish" and "Inquisition"; the first
four have (the default) value of 1.0 and the last has a value of 5.0.

The attributes that appear in the domain aren't defined in any headers or even
separate files, as with other formats supported by Orange.

If attribute appears more than once, its values are added. For instance, the
value of attribute "surprise" in the second examples is 6.0 and the value of
"fear" is 2.0; the former appears three times with values of 3.0, 2.0 and 1.0,
and the latter appears twice with value of 1.0.

All attributes are loaded as optional meta-attributes, so zero values don't
take any memory (unless they are given, but initialized to zero). See also
section on :ref:`meta attributes <meta-attributes>` in the reference for domain
descriptors.

Notice that at the time of writing this reference only association rules can
directly use examples presented in the basket format.
