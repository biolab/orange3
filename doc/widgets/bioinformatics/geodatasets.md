GEO Data Sets
=============

![GEO Data Sets widget icon](icons/geo-data-sets.png)

Provides access to data sets from gene expression omnibus ([GEO
DataSets](http://www.ncbi.nlm.nih.gov/gds)).

Signals
-------

**Inputs**:

-   (None)

**Outputs**:

- **Data**

  Attribute-valued data set selected in the widget.

Description
-----------

**[GEO DataSets](http://www.ncbi.nlm.nih.gov/gds)** is a data base of gene
expression curated profiles maintained by [NCBI](http://www.ncbi.nlm.nih.gov/) and included in the [Gene
Expression Omnibus](http://www.ncbi.nlm.nih.gov/geo/info/datasets.html). This Orange widget provides
access to all its data sets and outputs a data set selected for further
processing. For convenience, each data set is stored locally. 
Upon the user's first access the widget loads the
data from GEO, but then, in any further queries, supports the offline
access.

![GEO Data Sets widget](images/GEOdataset-stamped.png)

1. Information on the GEO data set collection. Cached data sets are the ones currently stored in your local space.
2. Set output features. If *Genes or spots* is selected, genes (or spots) will be used as attributes. Alternatively samples
   will be used as attributes. *Merge spots of same gene* allows you to merge data from the same gene. Finally, in the
   *Data set name* you can rename the output data. GEO title will be used as a default name.
3. If *Auto commit is on*, then the selected data set will be automatically communicated to other widgets. Alternatively,
   click *Commit*.
4. *Filter* allows you to search for the data set. Below you see a list of GEO data sets with an ID number (link to the NCBI
   Data Set Browser), title,
   organism used in the experiment, number of samples, features, genes, subsets and a reference number for the PubMed
   journal (link to the article abstract).
5. Short description of the experiment from which the data set is sourced.
6. Select which *Sample Annotations* will be used in the output.

Example
-------

**GEO Data Sets** is a simple widget that works in a similar way as **File** widget. In the example below
we selected *Caffeine effect: time dose and response* data set from the GEO data base and used *Genes or spots* as
attributes. We inspected the data in *Data Table*. Then, to narrow down the number of attributes, we selected
3 genes in the **Select Columns** widget for a detailed analysis in another data table.

<img src="images/GEODataSets-Example2.png" alt="image" width="600">
