GEO Data Sets
=============

![GEO Data Sets widget icon]

Provides access to data sets from gene expression omnibus ([GEO
DataSets]).

Signals
-------

Inputs:

:   -   (None)

Outputs:

:   -   

        Data

        :   Attribute-valued data set created in the widget.

Description
-----------

Data sets from GEO (officially, [GEO DataSets]) is a data base of gene
expression curated profiles maintained by [NCBI] and included in [Gene
Expression Omnibus][GEO DataSets]). This Orange widget provides an
access to all its data sets and outputs a data table that can be further
mined in Orange. For convenience, each data set selected for further
processing is stored locally. Upon its first access the widget loads the
data from GEO, but then, in any further queries, supports the offline
access.

![GEO Data Sets widget]

Examples
--------

  [GEO Data Sets widget icon]: ../../orangecontrib/bio/widgets/icons/GEODataSets.svg
  [GEO DataSets]: http://www.ncbi.nlm.nih.gov/geo/
  [NCBI]: http://www.ncbi.nlm.nih.gov/
  [GEO Data Sets widget]: images/geodatasets-stamped.png