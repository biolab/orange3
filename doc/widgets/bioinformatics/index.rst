Orange Bioinformatics documentation
===================================

Orange Bioinformatics is an add-on for Orange_ data mining software package. It
extends Orange by providing functionality for some elementary tasks in
bioinformatics, like gene set analysis, enrichment, and access to pathway
libraries. Included are also widgets for with graphical user interface
for, mainly, gene expression analytics.

.. _Orange: http://orange.biolab.si/

Widgets
-------

.. toctree::
   :maxdepth: 1
 
   PIPA.rst
   databases.rst
   geodatasets.rst

Scripting Reference
-------------------
   
.. toctree::
   :maxdepth: 1

   reference/arrayexpress.rst
   reference/biomart.rst
   reference/dicty.rst
   reference/dictybase.rst
   reference/gene.rst
   reference/gene.homology.rst
   reference/genesets.rst
   reference/geo.rst
   reference/go.rst
   reference/gsea.rst
   reference/kegg.rst
   reference/omim.rst
   reference/ontology.rst
   reference/ppi.rst
   reference/taxonomy.rst
   reference/utils.stats.rst

Installation
------------

To install Bioinformatics add-on for Orange from PyPi_ run::

    pip install Orange-Bioinformatics

To install it from source code run::

    python setup.py install

To build Python egg run::

    python setup.py bdist_egg

To install add-on in `development mode`_ run::

    python setup.py develop

.. _development mode: http://packages.python.org/distribute/setuptools.html#development-mode
.. _PyPi: http://pypi.python.org/pypi

Source Code and Issue Tracker
-----------------------------

Source code is available on Bitbucket_. For issues and wiki we use Trac_.

.. _Bitbucket: https://bitbucket.org/biolab/orange-bioinformatics
.. _Trac: http://orange.biolab.si/trac/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
